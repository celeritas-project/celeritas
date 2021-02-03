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

    // Reserve host space (MUST reserve subshells and transition data to avoid
    // invalidating spans).
    size_type ss_size = 0;
    size_type id_size = 0;
    size_type tr_size = 0;
    for (const auto& el : inp.elements)
    {
        ss_size += el.fluor.size();
        for (size_type i = 0; i < el.fluor.size(); ++i)
        {
            id_size += el.fluor[i].initial_shell.size();
            tr_size += el.fluor[i].transition_energy.size()
                       + el.fluor[i].transition_prob.size();
            if (is_auger_enabled_)
            {
                // If Auger is enabled we allocate space for Auger shells in
                // the radiative transitions too; they will just be flagged as
                // unassigned
                id_size += el.auger[i].initial_shell.size()
                           + el.auger[i].auger_shell.size()
                           + el.fluor[i].initial_shell.size();
                tr_size += el.auger[i].transition_energy.size()
                           + el.auger[i].transition_prob.size();
            }
        }
    }
    host_elements_.reserve(inp.elements.size());
    host_shells_.reserve(ss_size);
    host_id_data_.reserve(id_size);
    host_tr_data_.reserve(tr_size);

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
        device_id_data_ = DeviceVector<size_type>{host_id_data_.size()};
        device_tr_data_ = DeviceVector<real_type>{host_tr_data_.size()};

        // Remap shell->data spans
        auto remap_id = make_span_remapper(make_span(host_id_data_),
                                           device_id_data_.device_pointers());
        auto remap_tr = make_span_remapper(make_span(host_tr_data_),
                                           device_tr_data_.device_pointers());
        std::vector<AtomicRelaxSubshell> temp_device_shells = host_shells_;
        for (AtomicRelaxSubshell& ss : temp_device_shells)
        {
            ss.transition_energy = remap_tr(ss.transition_energy);
            ss.transition_prob   = remap_tr(ss.transition_prob);
            ss.initial_shell     = remap_id(ss.initial_shell);
            if (is_auger_enabled_)
            {
                ss.auger_shell = remap_id(ss.auger_shell);
            }
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
        device_id_data_.copy_to_device(make_span(host_id_data_));
        device_tr_data_.copy_to_device(make_span(host_tr_data_));
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
    result.unassigned  = unassigned();

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
    result.unassigned  = unassigned();

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
        result.max_secondary = std::pow(2, inp.auger.size()) - 1;
    }
    else
    {
        result.max_secondary = inp.fluor.size();
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
    CELER_EXPECT(inp.designators.size() == inp.fluor.size());
    CELER_EXPECT(host_shells_.size() + inp.fluor.size()
                 <= host_shells_.capacity());

    // Allocate subshells
    auto start = host_shells_.size();
    host_shells_.resize(start + inp.fluor.size());
    Span<AtomicRelaxSubshell> result{host_shells_.data() + start,
                                     inp.fluor.size()};

    for (auto i : range(inp.fluor.size()))
    {
        // Store the radiative transition energies and append the non-radiative
        // energies if Auger effect is enabled
        result[i].transition_energy
            = extend(inp.fluor[i].transition_energy, &host_tr_data_);
        if (is_auger_enabled_)
        {
            auto ext = extend(inp.auger[i].transition_energy, &host_tr_data_);
            result[i].transition_energy
                = {result[i].transition_energy.begin(), ext.end()};
        }

        // Store the radiative transition probabilities and append the
        // non-radiative probabilities if Auger effect is enabled
        result[i].transition_prob
            = extend(inp.fluor[i].transition_prob, &host_tr_data_);
        if (is_auger_enabled_)
        {
            auto ext = extend(inp.auger[i].transition_prob, &host_tr_data_);
            result[i].transition_prob
                = {result[i].transition_prob.begin(), ext.end()};

            // For a given subshell vacancy, EADL transition probabilities are
            // normalized so that the sum over all radiative and non-radiative
            // transitions is 1
            real_type norm = std::accumulate(result[i].transition_prob.begin(),
                                             result[i].transition_prob.end(),
                                             real_type(0));
            CELER_ASSERT(soft_near(1., norm, 1.e-5));
        }

        // Map the shell designators to the indices in the shells array
        std::unordered_map<size_type, size_type> des_to_id;
        for (auto i : range(inp.designators.size()))
        {
            des_to_id[inp.designators[i]] = i;
        }
        CELER_ASSERT(des_to_id.size() == inp.designators.size());

        // Convert the initial shell designators to indices and store
        std::vector<size_type> shell_id;
        map_des_to_id(des_to_id, inp.fluor[i].initial_shell, &shell_id);
        result[i].initial_shell = extend(shell_id, &host_id_data_);
        if (is_auger_enabled_)
        {
            map_des_to_id(des_to_id, inp.auger[i].initial_shell, &shell_id);
            auto ext = extend(shell_id, &host_id_data_);
            result[i].initial_shell
                = {result[i].initial_shell.begin(), ext.end()};

            // Convert the Auger shell designators to indices and store. For
            // radiative transitions, mark the Auger shells as unassigned
            shell_id.resize(inp.fluor[i].initial_shell.size());
            std::fill(shell_id.begin(), shell_id.end(), unassigned());
            result[i].auger_shell = extend(shell_id, &host_id_data_);

            map_des_to_id(des_to_id, inp.auger[i].auger_shell, &shell_id);
            ext                   = extend(shell_id, &host_id_data_);
            result[i].auger_shell = {result[i].auger_shell.begin(), ext.end()};
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Map subshell designators to indices in the shells array
 */
void AtomicRelaxationParams::map_des_to_id(
    const std::unordered_map<size_type, size_type>& des_to_id,
    const std::vector<size_type>&                   des,
    std::vector<size_type>*                         id)
{
    id->clear();
    for (auto d : des)
    {
        auto it = des_to_id.find(d);
        if (it == des_to_id.end())
        {
            // There is no transition data for this shell
            id->push_back(unassigned());
        }
        else
        {
            id->push_back(it->second);
        }
    }
    CELER_ENSURE(id->size() == des.size());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
