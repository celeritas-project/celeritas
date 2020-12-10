//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceVector.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a number of allocated elements.
 */
template<class T>
DeviceVector<T>::DeviceVector(size_type count)
    : allocation_(count * sizeof(T)), size_(count), capacity_(count)
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
    swap(capacity_, other.capacity_);
    swap(allocation_, other.allocation_);
}

//---------------------------------------------------------------------------//
/*!
 * Change the size without changing capacity. There is no reallocation of
 * storage: the vector can only shrink or grow up to the container capacity.
 */
template<class T>
void DeviceVector<T>::resize(size_type size)
{
    REQUIRE(size <= this->capacity());
    size_ = size;
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 */
template<class T>
void DeviceVector<T>::copy_to_device(constSpan_t data)
{
    REQUIRE(data.size() == this->size());
    allocation_.copy_to_device(
        {reinterpret_cast<const byte*>(data.data()), data.size() * sizeof(T)});
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to host.
 */
template<class T>
void DeviceVector<T>::copy_to_host(Span_t data) const
{
    REQUIRE(data.size() == this->size());
    allocation_.copy_to_host(
        {reinterpret_cast<byte*>(data.data()), data.size() * sizeof(T)});
}

//---------------------------------------------------------------------------//
/*!
 * Get an on-device view to the data.
 */
template<class T>
auto DeviceVector<T>::device_pointers() -> Span_t
{
    return {reinterpret_cast<T*>(allocation_.device_pointers().data()),
            this->size()};
}

//---------------------------------------------------------------------------//
/*!
 * Get an on-device view to the data.
 */
template<class T>
auto DeviceVector<T>::device_pointers() const -> constSpan_t
{
    return {reinterpret_cast<const T*>(allocation_.device_pointers().data()),
            this->size()};
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
} // namespace celeritas
