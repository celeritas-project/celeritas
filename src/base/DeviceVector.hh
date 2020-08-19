//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceVector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include "DeviceAllocation.hh"
#include "detail/InitializedValue.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Host-compiler-friendly vector for uninitialized device-storage data.
 *
 * This class does *not* perform initialization on the data. The host code
 * must define and copy over suitable data.
 *
 * \code
    DeviceVector<double> myvec(100);
    myvec.copy_to_device(make_span(hostvec));
    myvec.copy_to_host(make_span(hostvec));
   \endcode
 */
template<class T>
class DeviceVector
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "DeviceVector element is not trivially copyable");

  public:
    //@{
    //! Type aliases
    using value_type  = T;
    using DevicePointers      = DeviceValue<span<value_type>>;
    using HostPointers        = HostValue<span<value_type>>;
    using constDevicePointers = DeviceValue<span<const value_type>>;
    using constHostPointers   = HostValue<span<const value_type>>;
    //@}

  public:
    // Construct with no elements
    DeviceVector() = default;

    // Construct with a number of elements
    DeviceVector(size_type num_bytes);

    // Swap with another vector
    inline void swap(DeviceVector& other) noexcept;

    // >>> ACCESSORS

    //! Get the number of elements allocated
    size_type size() const { return size_; }

    //! Whether any elements are stored
    bool empty() const { return size_ == 0; }

    // >>> DEVICE ACCESSORS

    // Copy data to device
    inline void copy_to_device(constHostPointers host_data);

    // Copy data to host
    inline void copy_to_host(HostPointers host_data) const;

    // Get a mutable view to device data
    inline DevicePointers device_pointers();

    // Get a const view to device data
    inline constDevicePointers device_pointers() const;

  private:
    DeviceAllocation allocation_;
    detail::InitializedValue<size_type> size_;
};

// Swap two vectors
template<class T>
inline void swap(DeviceVector<T>&, DeviceVector<T>&) noexcept;

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "DeviceVector.i.hh"

//---------------------------------------------------------------------------//
