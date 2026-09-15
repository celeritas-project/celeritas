//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/detail/NotImplementedField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Dummy class for map magnetic fields when no backend is available.
 *
 * \tparam ParamsData  The params data template for this field type
 *                     (e.g. CartMapFieldParamsData, RZMapFieldParamsData).
 *                     Must be a complete type when this class is instantiated.
 */
template<template<Ownership, MemSpace> class ParamsData>
class NotImplementedField
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = float;
    using Real3 = Array<celeritas::real_type, 3>;
    using ParamsRef = NativeCRef<ParamsData>;
    //!@}

  public:
    // Construct with the shared map data
    CELER_FUNCTION explicit NotImplementedField(ParamsRef const&)
    {
        CELER_NOT_CONFIGURED("covfie");
    }

    // Evaluate the magnetic field value for the given position
    CELER_FUNCTION Real3 operator()(Real3 const&) const
    {
        CELER_NOT_CONFIGURED("covfie");
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
