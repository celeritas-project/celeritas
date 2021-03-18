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
#include "Span.hh"
#include "detail/InitializedValue.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Host-compiler-friendly vector for uninitialized device-storage data.
 *
 * The device vector's allocation is fixed on construction, but can be resized
 * to a logically smaller space, or assigned (replaced) by a new DeviceVector.
 * As a consequence, no \c resize operation will invalidate \c Span references
 * to device data.
 *
 * This class does *not* perform initialization on the data. The host code must
 * define and copy over suitable data. For more complex data usage (dynamic
 * size increases and assignment without memory reallocation), use \c
 * thrust::device_vector.
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

    static_assert(std::is_trivially_destructible<T>::value,
                  "DeviceVector element is not trivially destructible");

  public:
    //!@{
    //! Type aliases
    using value_type = T;
    using SpanT      = Span<T>;
    using SpanConstT = Span<const T>;
    //!@}

  public:
    // Construct with no elements
    DeviceVector() = default;

    // Construct with a number of elements
    explicit DeviceVector(size_type count);

    // Swap with another vector
    inline void swap(DeviceVector& other) noexcept;

    // Change the size without changing capacity
    inline void resize(size_type size);

    //// ACCESSORS ////

    //! Get the number of elements
    size_type size() const { return size_; }

    //! Get the number of elements that can fit in the allocated storage
    size_type capacity() const { return capacity_; }

    //! Whether any elements are stored
    bool empty() const { return size_ == 0; }

    //// DEVICE ACCESSORS ////

    // Copy data to device
    inline void copy_to_device(SpanConstT host_data);

    // Copy data to host
    inline void copy_to_host(SpanT host_data) const;

    // Get a mutable view to device data
    SpanT device_pointers() { return {this->data(), this->size()}; }

    // Get a const view to device data
    SpanConstT device_pointers() const { return {this->data(), this->size()}; }

    // Raw pointer to device data (dangerous!)
    inline T* data();

    // Raw pointer to device data (dangerous!)
    inline const T* data() const;

  private:
    DeviceAllocation                    allocation_;
    detail::InitializedValue<size_type> size_;
    detail::InitializedValue<size_type> capacity_;
};

// Swap two vectors
template<class T>
inline void swap(DeviceVector<T>&, DeviceVector<T>&) noexcept;

//---------------------------------------------------------------------------//
/*!
 * Prevent accidental construction of Span from a device vector.
 *
 * Use \c dv.device_pointers() to get a span.
 */
template<class T>
CELER_FUNCTION Span<const T> make_span(const DeviceVector<T>& dv)
{
    static_assert(sizeof(T) == 0, "Cannot 'make_span' from a device vector");
    return {dv.data(), dv.size()};
}

//---------------------------------------------------------------------------//
//! Prevent accidental construction of Span from a device vector.
template<class T>
CELER_FUNCTION Span<T> make_span(DeviceVector<T>& dv)
{
    static_assert(sizeof(T) == 0, "Cannot 'make_span' from a device vector");
    return {dv.data(), dv.size()};
}

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "DeviceVector.i.hh"
