//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/DeviceVector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "DeviceAllocation.hh"
#include "corecel/cont/Span.hh"
#include "detail/InitializedValue.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Host-compiler-friendly vector for uninitialized device-storage data.
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
#if !CELERITAS_USE_HIP
    // rocrand states have nontrivial destructors, and some HIP integer types
    // are not trivially copyable
    static_assert(std::is_trivially_copyable<T>::value,
                  "DeviceVector element is not trivially copyable");

    static_assert(std::is_trivially_destructible<T>::value,
                  "DeviceVector element is not trivially destructible");
#endif

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

    //// ACCESSORS ////

    //! Get the number of elements
    size_type size() const { return size_; }

    //! Whether any elements are stored
    bool empty() const { return size_ == 0; }

    //// DEVICE ACCESSORS ////

    // Copy data to device
    inline void copy_to_device(SpanConstT host_data);

    // Copy data to host
    inline void copy_to_host(SpanT host_data) const;

    // Get a mutable view to device data
    SpanT device_ref() { return {this->data(), this->size()}; }

    // Get a const view to device data
    SpanConstT device_ref() const { return {this->data(), this->size()}; }

    // Raw pointer to device data (dangerous!)
    inline T* data();

    // Raw pointer to device data (dangerous!)
    inline const T* data() const;

  private:
    DeviceAllocation                    allocation_;
    detail::InitializedValue<size_type> size_;
};

// Swap two vectors.
template<class T>
inline void swap(DeviceVector<T>& a, DeviceVector<T>& b) noexcept;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a number of allocated elements.
 */
template<class T>
DeviceVector<T>::DeviceVector(size_type count)
    : allocation_(count * sizeof(T)), size_(count)
{
}

//---------------------------------------------------------------------------//
/*!
 * Get the device data pointer.
 */
template<class T>
void DeviceVector<T>::swap(DeviceVector& other) noexcept
{
    using std::swap;
    swap(size_, other.size_);
    swap(allocation_, other.allocation_);
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 */
template<class T>
void DeviceVector<T>::copy_to_device(SpanConstT data)
{
    CELER_EXPECT(data.size() == this->size());
    allocation_.copy_to_device(
        {reinterpret_cast<const Byte*>(data.data()), data.size() * sizeof(T)});
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to host.
 */
template<class T>
void DeviceVector<T>::copy_to_host(SpanT data) const
{
    CELER_EXPECT(data.size() == this->size());
    allocation_.copy_to_host(
        {reinterpret_cast<Byte*>(data.data()), data.size() * sizeof(T)});
}

//---------------------------------------------------------------------------//
/*!
 * Get a device data pointer.
 */
template<class T>
T* DeviceVector<T>::data()
{
    return reinterpret_cast<T*>(allocation_.device_ref().data());
}

//---------------------------------------------------------------------------//
/*!
 * Get a device data pointer.
 */
template<class T>
const T* DeviceVector<T>::data() const
{
    return reinterpret_cast<const T*>(allocation_.device_ref().data());
}

//---------------------------------------------------------------------------//
/*!
 * Swap two vectors.
 */
template<class T>
void swap(DeviceVector<T>& a, DeviceVector<T>& b) noexcept
{
    return a.swap(b);
}

//---------------------------------------------------------------------------//
/*!
 * Prevent accidental construction of Span from a device vector.
 *
 * Use \c dv.device_ref() to get a span.
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
