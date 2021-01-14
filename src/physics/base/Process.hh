//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Process.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include "Applicability.hh"
#include "ModelIdGenerator.hh"
#include "Types.hh"
#include "ValueGridBuilder.hh"

namespace celeritas
{
class Model;
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
 */
class Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstModel       = std::shared_ptr<const Model>;
    using UPConstGridBuilder = std::unique_ptr<const ValueGridBuilder>;
    using VecModel           = std::vector<SPConstModel>;
    //!@}

    //! Array builders for along-step energy loss and other quantities
    struct StepLimitBuilders
    {
        UPConstGridBuilder macro_xs;    //!< Cross section [1/cm]
        UPConstGridBuilder energy_loss; //!< dE/dx [MeV/cm]
        UPConstGridBuilder range;       //!< Range limit [cm]
    };

  public:
    // Virtual destructor for polymorphic deletion
    virtual ~Process() = 0;

    //! Construct the models associated with this process
    virtual VecModel build_models(ModelIdGenerator next_id) const = 0;

    //! Get the interaction cross sections for the given energy range
    virtual StepLimitBuilders step_limits(Applicability range) const = 0;

    //! Name of the process
    virtual std::string label() const = 0;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
