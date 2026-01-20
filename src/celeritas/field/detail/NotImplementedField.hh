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
#include "celeritas/field/CartMapFieldData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Dummy class for cartesian map magnetic field when no backend is available.
 */
class NotImplementedField
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = float;
    using Real3 = Array<celeritas::real_type, 3>;
    using ParamsRef = NativeCRef<CartMapFieldParamsData>;
    //!@}

  public:
    // Construct with the shared map data
    inline CELER_FUNCTION explicit NotImplementedField(ParamsRef const&);

    // Evaluate the magnetic field value for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 const&) const;
};

CELER_FUNCTION
NotImplementedField::NotImplementedField(ParamsRef const&)
{
    CELER_NOT_CONFIGURED("covfie");
}

CELER_FUNCTION auto NotImplementedField::operator()(Real3 const&) const -> Real3
{
    CELER_NOT_CONFIGURED("covfie");
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
