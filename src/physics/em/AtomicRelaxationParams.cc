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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a vector of element identifiers.
 *
 * \note The EADL only provides transition probabilities for 6 <= Z <= 100, so
 * there will be no atomic relaxation data for Z < 6.
 */
AtomicRelaxationParams::AtomicRelaxationParams(const Input& inp)
    : is_auger_enabled_(inp.is_auger_enabled)
    , electron_id_(inp.electron_id)
    , gamma_id_(inp.gamma_id)
{
    CELER_EXPECT(!inp.elements.empty());
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

    // Build elements
    for (const auto& el : inp.elements)
    {
        this->append_element(el);
    }

    if (celeritas::is_device_enabled())
    {
        // Allocate device vectors
        device_elements_
            = DeviceVector<AtomicRelaxElement>{host_elements_.size()};
        device_shells_ = DeviceVector<AtomicRelaxSubshell>{host_shells_.size()};
        device_transitions_
            = DeviceVector<AtomicRelaxTransition>{host_transitions_.size()};

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
    result.elements    = make_span(host_elements_);
    result.electron_id = electron_id_;
    result.gamma_id    = gamma_id_;

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Access EADL data on the device.
 */
AtomicRelaxParamsPointers AtomicRelaxationParams::device_pointers() const
{
    CELER_EXPECT(celeritas::is_device_enabled());

    AtomicRelaxParamsPointers result;
    result.elements    = device_elements_.device_pointers();
    result.electron_id = electron_id_;
    result.gamma_id    = gamma_id_;

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION
//---------------------------------------------------------------------------//
/*!
 * Convert an element input to a AtomicRelaxElement and store.
 */
void AtomicRelaxationParams::append_element(const ElementInput& inp)
{
    AtomicRelaxElement result;

    // TODO: For an element Z with n shells of transition data, you can bound
    // this worst case as n for radiative transitions and 2^n - 1 for
    // non-radiative transitions if for a given vacancy the transitions always
    // originate from the next subshell up. Physically this won't happen, so
    // can we bound this tighter (maybe O(100) for non-radiative transitions,
    // but that's still a lot)? Can we impose an energy cut below which we
    // won't create secondaries in the relaxation cascade to reduce it further?
    if (is_auger_enabled_)
    {
        result.max_secondary = std::exp2(inp.shells.size()) - 1;
    }
    else
    {
        result.max_secondary = inp.shells.size();
    }

    // Copy subshell transition data
    result.shells = this->extend_shells(inp);

    // Add to host vector
    host_elements_.push_back(result);
}

//---------------------------------------------------------------------------//
/*!
 * Process and store electron subshells to the internal list.
 */
Span<AtomicRelaxSubshell>
AtomicRelaxationParams::extend_shells(const ElementInput& inp)
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
    for (SubshellId::value_type i : range(inp.shells.size()))
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
    const std::vector<TransitionInput>& transitions)
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
