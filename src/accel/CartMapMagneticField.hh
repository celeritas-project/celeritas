//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CartMapMagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4MagneticField.hh>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/field/CartMapFieldParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Uniform grid specification for a single axis
struct Uniform
{
    G4double min{};  //!< Minimum coordinate value
    G4double max{};  //!< Maximum coordinate value
    size_type num{};  //!< Number of grid points

    //! Check if parameters are valid
    explicit operator bool() const { return max > min && num > 1; }
};

//---------------------------------------------------------------------------//
//! POD struct for CartMap field grid parameters
struct CartMapFieldGridParams
{
    Uniform x{};  //!< X-axis grid specification
    Uniform y{};  //!< Y-axis grid specification
    Uniform z{};  //!< Z-axis grid specification

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

  private:
    // Forward declaration for pImpl
    struct Impl;

    // Custom deleter for pImpl
    struct ImplDeleter
    {
        void operator()(Impl* ptr) const;
    };

  public:
    // Construct with CartMapFieldParams
    explicit CartMapMagneticField(SPConstFieldParams field_params);

    // Default move semantics work with custom deleter
    CELER_DEFAULT_MOVE_DELETE_COPY(CartMapMagneticField);

    // Destructor
    ~CartMapMagneticField() override = default;

    // Calculate values of the magnetic field vector
    void GetFieldValue(G4double const point[3], G4double* field) const override;

  private:
    std::unique_ptr<Impl, ImplDeleter> pimpl_;
};

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
}  // namespace celeritas
