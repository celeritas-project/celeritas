//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#if CELERITAS_USE_COVFIE
#    include "detail/CartMapField.covfie.hh"
#else

#    include "corecel/Assert.hh"
#    include "corecel/Macros.hh"
#    include "corecel/Types.hh"
#    include "corecel/cont/Array.hh"

#    include "CartMapFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Dummy class for cartesian map magnetic field when no backend is available.
 */
class CartMapField
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = float;
    using Real3 = Array<celeritas::real_type, 3>;
    using FieldParamsRef
        = CartMapFieldParamsData<Ownership::const_reference, MemSpace::native>;
    using field_view_t = void;
    //!@}

  public:
    // Construct with the shared map data
    inline CELER_FUNCTION explicit CartMapField(FieldParamsRef const&);

    // Evaluate the magnetic field value for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 const&) const;
};

CELER_FUNCTION
CartMapField::CartMapField(FieldParamsRef const&)
{
    CELER_NOT_CONFIGURED("Covfie");
}

CELER_FUNCTION auto CartMapField::operator()(Real3 const&) const -> Real3
{
    CELER_NOT_CONFIGURED("Covfie");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // CELERITAS_USE_COVFIE
