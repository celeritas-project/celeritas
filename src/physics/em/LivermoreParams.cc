//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermoreParams.cc
//---------------------------------------------------------------------------//
#include "LivermoreParams.hh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include "base/Range.hh"
#include "base/SoftEqual.hh"
#include "base/SpanRemapper.hh"
#include "comm/Device.hh"
#include "comm/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a vector of element identifiers.
 */
LivermoreParams::LivermoreParams(Input inp)
{
    REQUIRE(!inp.elements.empty());

    // Sort elements by increasing element ID
    std::sort(inp.elements.begin(),
              inp.elements.end(),
              [](const ElementInput& lhs, const ElementInput& rhs) {
                  return lhs.el_id < rhs.el_id;
              });

    // Reserve host space (MUST reserve subshells and cross section data to
    // avoid invalidating spans).
    size_type subshell_size = 0;
    size_type data_size     = 0;
    for (const auto& el : inp.elements)
    {
        subshell_size += el.shells.size();
        data_size += el.xs_low.energy.size() + el.xs_low.xs_eloss.size()
                     + el.xs_high.energy.size() + el.xs_high.xs_eloss.size();

        for (const auto& shell : el.shells)
        {
            data_size += shell.param_low.size() + shell.param_high.size()
                         + shell.xs.size() + shell.energy.size();
        }
    }
    host_elements_.reserve(inp.elements.size());
    host_shells_.reserve(subshell_size);
    host_data_.reserve(data_size);

    // Build elements
    for (const auto& el : inp.elements)
    {
        this->append_livermore_element(el);
    }

    if (celeritas::is_device_enabled())
    {
        // Allocate device vectors
        device_elements_
            = DeviceVector<LivermoreElement>{host_elements_.size()};
        device_shells_ = DeviceVector<LivermoreSubshell>{host_shells_.size()};
        device_data_   = DeviceVector<real_type>{host_data_.size()};

        // Remap shell->data spans
        auto remap_data = make_span_remapper(make_span(host_data_),
                                             device_data_.device_pointers());
        std::vector<LivermoreSubshell> temp_device_shells = host_shells_;
        for (LivermoreSubshell& shell : temp_device_shells)
        {
            shell.xs.energy  = remap_data(shell.xs.energy);
            shell.xs.xs      = remap_data(shell.xs.xs);
            shell.param_low  = remap_data(shell.param_low);
            shell.param_high = remap_data(shell.param_high);
        }

        // Remap element->shell spans and element->data spans
        auto remap_shells = make_span_remapper(
            make_span(host_shells_), device_shells_.device_pointers());
        std::vector<LivermoreElement> temp_device_elements = host_elements_;
        for (LivermoreElement& el : temp_device_elements)
        {
            el.xs_low.energy  = remap_data(el.xs_low.energy);
            el.xs_low.xs      = remap_data(el.xs_low.xs);
            el.xs_high.energy = remap_data(el.xs_high.energy);
            el.xs_high.xs     = remap_data(el.xs_high.xs);
            el.shells         = remap_shells(el.shells);
        }

        // Copy vectors to device
        device_elements_.copy_to_device(make_span(temp_device_elements));
        device_shells_.copy_to_device(make_span(temp_device_shells));
        device_data_.copy_to_device(make_span(host_data_));
    }

    ENSURE(host_elements_.size() == inp.elements.size());
    ENSURE(host_shells_.size() <= host_shells_.capacity());
    ENSURE(host_data_.size() <= host_data_.capacity());
}

//---------------------------------------------------------------------------//
/*!
 * Access Livermore data on the host.
 */
LivermoreParamsPointers LivermoreParams::host_pointers() const
{
    LivermoreParamsPointers result;
    result.elements = make_span(host_elements_);

    ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Access Livermore data on the device.
 */
LivermoreParamsPointers LivermoreParams::device_pointers() const
{
    LivermoreParamsPointers result;
    result.elements = device_elements_.device_pointers();

    ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION
//---------------------------------------------------------------------------//
/*!
 * Convert an element input to a LivermoreElement and store.
 */
void LivermoreParams::append_livermore_element(const ElementInput& inp)
{
    LivermoreElement result;

    // Copy basic properties
    result.xs_low.energy  = this->extend_data(inp.xs_low.energy);
    result.xs_low.xs      = this->extend_data(inp.xs_low.xs_eloss);
    result.xs_low.interp  = Interp::linear;
    result.xs_high.energy = this->extend_data(inp.xs_high.energy);
    result.xs_high.xs     = this->extend_data(inp.xs_high.xs_eloss);
    result.xs_low.interp  = Interp::linear; // TODO: spline
    result.shells         = this->extend_shells(inp);
    result.thresh_low     = inp.thresh_low;
    result.thresh_high    = inp.thresh_high;

    // Add to host vector
    host_elements_.push_back(result);
}

//---------------------------------------------------------------------------//
/*!
 * Process and store electron subshells to the internal list.
 */
span<LivermoreSubshell> LivermoreParams::extend_shells(const ElementInput& inp)
{
    REQUIRE(host_shells_.size() + inp.shells.size() <= host_shells_.capacity());

    // Allocate subshells
    auto start_size = host_shells_.size();
    host_shells_.resize(start_size + inp.shells.size());
    span<LivermoreSubshell> result{host_shells_.data() + start_size,
                                   inp.shells.size()};

    // Store binding energy, fit parameters, and tabulated cross sections
    for (auto i : range(inp.shells.size()))
    {
        result[i].binding_energy = inp.shells[i].binding_energy;
        result[i].xs.energy      = this->extend_data(inp.shells[i].energy);
        result[i].xs.xs          = this->extend_data(inp.shells[i].xs);
        result[i].xs.interp      = Interp::linear;
        result[i].param_low      = this->extend_data(inp.shells[i].param_low);
        result[i].param_high     = this->extend_data(inp.shells[i].param_high);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process and store tabulated cross sections, energies, and fit parameters.
 */
span<real_type> LivermoreParams::extend_data(const std::vector<real_type>& data)
{
    REQUIRE(host_data_.size() + data.size() <= host_data_.capacity());

    // Allocate data
    host_data_.insert(host_data_.end(), data.begin(), data.end());
    return span<real_type>{host_data_.data() + host_data_.size() - data.size(),
                           data.size()};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
