//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/MockProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <vector>

#include "celeritas/Quantities.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Energy loss rate [MeV/cm] per volume [cm^-3] -> [MeV * cm^2]
struct MevCmSq
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return units::Mev::value() * ipow<2>(units::centimeter);
    }
};

using MevCmSqLossDens = RealQuantity<MevCmSq>;

//! Energy loss rate
struct MevPerCm
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return units::Mev::value() / units::centimeter;
    }
};

using MevPerCmLoss = RealQuantity<MevPerCm>;

//---------------------------------------------------------------------------//
/*!
 * Mock process.
 *
 * Multiple instances of this process can be created to test out the physics.
 * The value grids are all parameterized:
 * - Cross section is scaled by the material's atomic number density, and is
 *   constant with energy.
 * - Energy loss rate is also constant with energy and scales with the number
 *   density.
 * - Range is determined by the energy loss rate -- constant energy loss rate k
 *   means linear range of E/k.
 *
 * The given applicability vector has one element per model that it will
 * create. Each model can have a different particle type and/or energy range.
 */
class MockProcess : public Process
{
  public:
    //!@{
    //! \name Type aliases
    using BarnMicroXs = RealQuantity<units::Barn>;
    using VecApplicability = std::vector<Applicability>;
    using VecMicroXs = std::vector<BarnMicroXs>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using ModelCallback = std::function<void(ActionId)>;
    //!@}

    struct Input
    {
        SPConstMaterials materials;
        std::string label;
        bool supports_integral_xs{true};
        bool applies_at_rest{false};
        VecApplicability applic;  //!< Applicablity per model
        ModelCallback interact;  //!< MockModel::interact callback
        VecMicroXs xs;  //!< Constant per atom [bn]
        MevCmSqLossDens energy_loss{};  //!< Constant per atom
    };

  public:
    explicit MockProcess(Input data);

    VecModel build_models(ActionIdIter start_id) const final;
    StepLimitBuilders step_limits(Applicability range) const final;
    bool supports_integral_xs() const final;
    bool applies_at_rest() const final;
    std::string_view label() const final;

  private:
    Input data_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
