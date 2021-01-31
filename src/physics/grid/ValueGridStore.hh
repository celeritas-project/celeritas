//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/DeviceVector.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "XsGridInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage data and help construction of physics value grids.
 *
 * Currently this only constructs a single value grid datatype, the
 * XsGridPointers, but with this framework (virtual \c
 * ValueGridXsBuilder::build method taking an instance of this class) it can be
 * extended to build additional grid types as well.
 *
 * \code
    ValueGridStore store(2, 100);
    store.push_back(host_ptrs);
    store.push_back(host_ptrs);
    store.copy_to_device();
   \endcode
 */
class ValueGridStore
{
  public:
    //!@{
    //! Type aliases
    using ValueGridPointers = Span<const XsGridPointers>;
    //!@}

  public:
    // Construct without any capacity
    ValueGridStore() = default;

    // Construct with storage space requirements
    ValueGridStore(size_type num_grids, size_type num_values);

    // Add a grid of host pointer data
    void push_back(const XsGridPointers& host_pointers);

    // Add a grid of generic data
    void push_back(const GenericGridPointers& host_pointers);

    // Copy the data to device
    void copy_to_device();

    //// HOST ACCESSORS ////

    //! Number of constructed grids
    size_type size() { return host_xsgrids_.size() + host_grids_.size(); }

    //! Number of allocated grids
    size_type capacity() { return capacity_; }

    //! Get host data
    ValueGridPointers host_pointers() const;

    //// DEVICE ACCESSORS ////

    ValueGridPointers device_pointers() const;

  private:
    size_type                        capacity_{};
    std::vector<XsGridPointers>      host_xsgrids_;
    std::vector<GenericGridPointers> host_grids_;
    std::vector<real_type>           host_values_;

    DeviceVector<XsGridPointers> device_grids_;
    DeviceVector<real_type>      device_values_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
