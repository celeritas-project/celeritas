//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Process.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "corecel/cont/Range.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/ValueGridData.hh"

#include "Applicability.hh"

namespace celeritas
{
class Model;

//---------------------------------------------------------------------------//
/*!
 * Type of physics process.
 */
enum class ProcessType
{
    electromagnetic_discrete, //!< Discrete EM process
    electromagnetic_dedx,     //!< Continuous-discrete EM process
    electromagnetic_msc       //!< Multiple scattering
};

//---------------------------------------------------------------------------//
/*!
 * An interface/factory method for creating models.
 *
 * Currently processes pull their data from Geant4 which combines multiple
 * model cross sections into an individual range for each particle type.
 * Therefore we make the process responsible for providing the combined cross
 * section values -- currently this will use preprocessed Geant4 data but later
 * we could provide helper functions so that each Process can individually
 * combine its models.
 *
 * Each process has an interaction ("post step doit") and may have both energy
 * loss and range limiters.
 *
 * The StepLimitBuilders is a fixed-size array corresponding to the physics
 * interface enum \c ValueGridType :
 * - macro_xs:    Cross section [1/cm]
 * - energy_loss: dE/dx [MeV/cm]
 * - range:       Range limit [cm]
 */
class Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstModel       = std::shared_ptr<const Model>;
    using UPConstGridBuilder = std::unique_ptr<const ValueGridBuilder>;
    using VecModel           = std::vector<SPConstModel>;
    using StepLimitBuilders  = ValueGridArray<UPConstGridBuilder>;
    using ActionIdIter       = RangeIter<ActionId>;
    //!@}

  public:
    // Virtual destructor for polymorphic deletion
    virtual ~Process();

    //! Construct the models associated with this process
    virtual VecModel build_models(ActionIdIter start_id) const = 0;

    //! Get the interaction cross sections for the given energy range
    virtual StepLimitBuilders step_limits(Applicability range) const = 0;

    //! Type of process
    virtual ProcessType type() const = 0;

    //! Name of the process
    virtual std::string label() const = 0;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
