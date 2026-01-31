//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CartMapMagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/field/CartMapFieldInput.hh"
#include "celeritas/field/CartMapFieldParams.hh"
#include "celeritas/g4/MagneticField.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! POD struct for CartMap field grid parameters
struct CartMapFieldGridParams
{
    // TODO: should be double?
    AxisGrid<real_type> x{};  //!< X-axis grid specification
    AxisGrid<real_type> y{};  //!< Y-axis grid specification
    AxisGrid<real_type> z{};  //!< Z-axis grid specification

    //! Check if parameters are valid for field generation
    explicit operator bool() const { return x && y && z; }
};

//---------------------------------------------------------------------------//
// Generate field input with user-defined uniform grid and explicit field
CartMapFieldInput MakeCartMapFieldInput(G4Field const& field,
                                        CartMapFieldGridParams const& params);

// Generate field input with user-defined uniform grid from global field
CartMapFieldInput MakeCartMapFieldInput(CartMapFieldGridParams const& params);

//---------------------------------------------------------------------------//
/*!
 * On-the-fly field calculation with covfie using Celeritas data+units.
 *
 * This "adapter" implementation hides the covfie dependency from downstream
 * users.
 */
struct CartAdapterField
{
    HostCRef<CartMapFieldParamsData> const& data;

    Real3 operator()(Real3 const&) const;
};

//---------------------------------------------------------------------------//
//! Geant4 magnetic field class for XYZ uniform grid field
using CartMapMagneticField
    = celeritas::MagneticField<CartMapFieldParams, CartAdapterField>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
