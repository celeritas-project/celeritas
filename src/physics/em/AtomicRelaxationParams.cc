//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxationParams.cc
//---------------------------------------------------------------------------//
#include "AtomicRelaxationParams.hh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include "base/Range.hh"
#include "base/SoftEqual.hh"
#include "base/SpanRemapper.hh"
#include "base/VectorUtils.hh"
#include "comm/Device.hh"
#include "detail/Utils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a vector of element identifiers.
 *
 * \note The EADL only provides transition probabilities for 6 <= Z <= 100, so
 * there will be no atomic relaxation data for Z < 6. Transitions are only
 * provided for K, L, M, N, and some O shells.
 */
AtomicRelaxationParams::AtomicRelaxationParams(const Input& inp)
    : is_auger_enabled_(inp.is_auger_enabled)
    , electron_id_(inp.particles->find(pdg::electron()))
    , gamma_id_(inp.particles->find(pdg::gamma()))
{
    CELER_EXPECT(inp.cutoffs);
    CELER_EXPECT(inp.materials);
    CELER_EXPECT(inp.particles);
    CELER_EXPECT(inp.elements.size() == inp.materials->num_elements());
    CELER_EXPECT(electron_id_);
    CELER_EXPECT(gamma_id_);

    // Reserve host space (MUST reserve subshells and transitions to avoid
    // invalidating spans).
    size_type ss_size = 0;
    size_type tr_size = 0;
    for (const auto& el : inp.elements)
    {
        ss_size += el.shells.size();
        for (const auto& shell : el.shells)
        {
            tr_size += shell.fluor.size();
            if (is_auger_enabled_)
            {
                tr_size += shell.auger.size();
            }
        }
    }
    host_elements_.reserve(inp.elements.size());
    host_shells_.reserve(ss_size);
    host_transitions_.reserve(tr_size);

    // Find the minimum electron and photon cutoff energy for each element over
    // all materials. This is used to calculate the maximum number of
    // secondaries that could be created in atomic relaxation for each element.
    size_type              num_elements = inp.materials->num_elements();
    std::vector<MevEnergy> min_ecut(num_elements, max_quantity());
    std::vector<MevEnergy> min_gcut(num_elements, max_quantity());
    for (auto mat_id : range(MaterialId{inp.materials->num_materials()}))
    {
        // Electron and photon energy cutoffs for this material
        auto cutoffs = inp.cutoffs->get(mat_id);
        auto ecut    = cutoffs.energy(electron_id_);
        auto gcut    = cutoffs.energy(gamma_id_);

        auto material = inp.materials->get(mat_id);
        for (auto comp_id : range(ElementComponentId{material.num_elements()}))
        {
            auto el_idx      = material.element_id(comp_id).get();
            min_ecut[el_idx] = min(min_ecut[el_idx], ecut);
            min_gcut[el_idx] = min(min_gcut[el_idx], gcut);
        }
    }

    // Build elements
    for (auto el_idx : range(num_elements))
    {
        this->append_element(
            inp.elements[el_idx], min_ecut[el_idx], min_gcut[el_idx]);
    }

    if (celeritas::device())
    {
        // Allocate device vectors
        device_elements_
            = DeviceVector<AtomicRelaxElement>(host_elements_.size());
        device_shells_ = DeviceVector<AtomicRelaxSubshell>(host_shells_.size());
        device_transitions_
            = DeviceVector<AtomicRelaxTransition>(host_transitions_.size());

        // Remap shell->transition spans
        auto remap_transitions
            = make_span_remapper(make_span(host_transitions_),
                                 device_transitions_.device_pointers());
        std::vector<AtomicRelaxSubshell> temp_device_shells = host_shells_;
        for (AtomicRelaxSubshell& ss : temp_device_shells)
        {
            ss.transitions = remap_transitions(ss.transitions);
        }

        // Remap element->shell spans
        auto remap_shells = make_span_remapper(
            make_span(host_shells_), device_shells_.device_pointers());
        std::vector<AtomicRelaxElement> temp_device_elements = host_elements_;
        for (AtomicRelaxElement& el : temp_device_elements)
        {
            el.shells = remap_shells(el.shells);
        }

        // Copy vectors to device
        device_elements_.copy_to_device(make_span(temp_device_elements));
        device_shells_.copy_to_device(make_span(temp_device_shells));
        device_transitions_.copy_to_device(make_span(host_transitions_));
    }

    CELER_ENSURE(host_elements_.size() == inp.elements.size());
}

//---------------------------------------------------------------------------//
/*!
 * Access EADL data on the host.
 */
AtomicRelaxParamsPointers AtomicRelaxationParams::host_pointers() const
{
    AtomicRelaxParamsPointers result;
    result.elements     = make_span(host_elements_);
    result.electron_id  = electron_id_;
    result.gamma_id     = gamma_id_;

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Access EADL data on the device.
 */
AtomicRelaxParamsPointers AtomicRelaxationParams::device_pointers() const
{
    CELER_EXPECT(celeritas::device());

    AtomicRelaxParamsPointers result;
    result.elements     = device_elements_.device_pointers();
    result.electron_id  = electron_id_;
    result.gamma_id     = gamma_id_;

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION
//---------------------------------------------------------------------------//
/*!
 * Convert an element input to a AtomicRelaxElement and store.
 */
void AtomicRelaxationParams::append_element(const ImportAtomicRelaxation& inp,
                                            MevEnergy electron_cutoff,
                                            MevEnergy gamma_cutoff)
{
    AtomicRelaxElement result;

    // Copy subshell transition data
    result.shells = this->extend_shells(inp);

    // Calculate the maximum possible number of secondaries that could be
    // created in atomic relaxation.
    result.max_secondary
        = detail::calc_max_secondaries(result, electron_cutoff, gamma_cutoff);

    // Maximum size of the stack used to store unprocessed vacancy subshell
    // IDs. For radiative transitions, there is only ever one vacancy waiting
    // to be processed. For non-radiative transitions, the upper bound on the
    // stack size is the number of shells that have transition data.
    result.max_stack_size = is_auger_enabled_ ? result.shells.size() : 1;

    // Add to host vector
    host_elements_.push_back(result);
}

//---------------------------------------------------------------------------//
/*!
 * Process and store electron subshells to the internal list.
 */
Span<AtomicRelaxSubshell>
AtomicRelaxationParams::extend_shells(const ImportAtomicRelaxation& inp)
{
    CELER_EXPECT(host_shells_.size() + inp.shells.size()
                 <= host_shells_.capacity());

    // Allocate subshells
    auto start = host_shells_.size();
    host_shells_.resize(start + inp.shells.size());
    Span<AtomicRelaxSubshell> result{host_shells_.data() + start,
                                     inp.shells.size()};

    // Create a mapping of subshell designator to index in the shells array
    des_to_id_.clear();
    for (SubshellId::size_type i : range(inp.shells.size()))
    {
        des_to_id_[inp.shells[i].designator] = SubshellId{i};
    }
    CELER_ASSERT(des_to_id_.size() == inp.shells.size());

    for (auto i : range(inp.shells.size()))
    {
        // Check that for a given subshell vacancy EADL transition
        // probabilities are normalized so that the sum over all radiative and
        // non-radiative transitions is 1
        real_type norm = 0.;
        for (const auto& transition : inp.shells[i].fluor)
            norm += transition.probability;
        for (const auto& transition : inp.shells[i].auger)
            norm += transition.probability;
        CELER_ASSERT(soft_equal(1., norm));

        // Store the radiative transitions
        auto fluor = this->extend_transitions(inp.shells[i].fluor);

        // Append the non-radiative transitions if Auger effect is enabled
        if (is_auger_enabled_)
        {
            auto auger = this->extend_transitions(inp.shells[i].auger);
            result[i].transitions = {fluor.begin(), auger.end()};
        }
        else
        {
            result[i].transitions = fluor;
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process and store transition data to the internal list.
 */
Span<AtomicRelaxTransition> AtomicRelaxationParams::extend_transitions(
    const std::vector<ImportAtomicTransition>& transitions)
{
    CELER_EXPECT(host_transitions_.size() + transitions.size()
                 <= host_transitions_.capacity());

    auto start = host_transitions_.size();
    host_transitions_.resize(start + transitions.size());

    for (auto i : range(transitions.size()))
    {
        auto& tr = host_transitions_[start + i];

        // Find the index in the shells array given the shell designator. If
        // the designator is not found, map it to an invalid value.
        tr.initial_shell = des_to_id_[transitions[i].initial_shell];
        tr.auger_shell   = des_to_id_[transitions[i].auger_shell];
        tr.probability   = transitions[i].probability;
        tr.energy        = transitions[i].energy;
    }

    return {host_transitions_.data() + start, transitions.size()};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
