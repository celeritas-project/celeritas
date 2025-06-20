//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CartMapMagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4MagneticField.hh>

#include "corecel/Types.hh"
#include "celeritas/field/CartMapFieldInput.hh"
#include "celeritas/field/CartMapFieldParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! POD struct for CartMap field grid parameters
struct CartMapFieldGridParams
{
    AxisGrid<real_type> x{};  //!< X-axis grid specification
    AxisGrid<real_type> y{};  //!< Y-axis grid specification
    AxisGrid<real_type> z{};  //!< Z-axis grid specification

    //! Check if parameters are valid for field generation
    explicit operator bool() const { return x && y && z; }
};

//---------------------------------------------------------------------------//
// Generate field input with user-defined uniform grid
CartMapFieldParams::Input
MakeCartMapFieldInput(CartMapFieldGridParams const& params);

//---------------------------------------------------------------------------//
/*!
 * A user magnetic field equivalent to celeritas::CartMapField.
 */
class CartMapMagneticField : public G4MagneticField
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstFieldParams = std::shared_ptr<CartMapFieldParams const>;
    //!@}

    // Construct with CartMapFieldParams
    explicit CartMapMagneticField(SPConstFieldParams field_params);

    // Calculate values of the magnetic field vector
    void GetFieldValue(G4double const point[3], G4double* field) const override;

  private:
    // Forward declaration for pImpl
    struct Impl;

    // Custom deleter for pImpl
    struct ImplDeleter
    {
        void operator()(Impl* ptr) const;
    };
    std::unique_ptr<Impl, ImplDeleter> pimpl_;
};

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
}  // namespace celeritas
