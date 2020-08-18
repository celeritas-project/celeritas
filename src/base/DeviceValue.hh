//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceValue.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Type-safe wrapper to indicate GPU/device storage.
 *
 * This should be used primarily to wrap \c Pointers classes returned from host
 * utility code (e.g. from a \c Store class).
 * \code
   struct FooStore
   {
      DeviceValue<FooPointers> device_pointers();
      DeviceValue<const FooPointers> device_pointers() const;
   };
   \endcode
 * used like
 * \code
   __global__ void do_stuff(const FooPointers pointers);

   void launch_kernel(const FooStore& foo)
   {
       do_stuff<<<...>>>(foo.device_pointers().value);
   }
   \endcode
 */
template<class T>
struct DeviceValue
{
    T value;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
