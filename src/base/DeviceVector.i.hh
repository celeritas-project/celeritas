//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
} // namespace celeritas
