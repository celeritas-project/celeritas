//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Model.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <set>
#include <string>  // IWYU pragma: export
#include <vector>

#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionInterface.hh"  // IWYU pragma: export
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/ValueGridData.hh"

#include "Applicability.hh"  // IWYU pragma: export

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Abstract base class representing a physics model with a discrete action.
 *
 * A Model is a representation (often an approximation) to a physics process
 * such as Compton scattering that is valid for one or more particle types in a
 * given range (or ranges) of energy.
 *
 * Each Model subclass is constructed with a unique ActionId by a Process,
 * which is effectively a group of Models. Once constructed, it is essentially
 * immutable.
 *
 * The model assumes a few responsibilities:
 * - It provides accessors for the ranges of applicability: the same model
 *   (interaction kernel) can apply to multiple particles at different energy
 *   ranges.
 * - It precalculates macroscopic cross sections for each range of
 *   applicability.
 * - It precalculates energy loss rates and range limiters for each range.
 * - If it has an interaction cross section, it provides an "execute" method
 *   for applying the interaction and possibly emitting secondaries.
 *
 * This class is similar to Geant4's G4VContinuousDiscrete process, but more
 * limited.
 */
class Model : public ExplicitActionInterface
{
  public:
    //@{
    //! Type aliases
    using UPConstGridBuilder = std::unique_ptr<ValueGridBuilder const>;
    using MicroXsBuilders = std::vector<UPConstGridBuilder>;
    using SetApplicability = std::set<Applicability>;
    //@}

  public:
    //! Get the applicable particle type and energy ranges of the model
    virtual SetApplicability applicability() const = 0;

    //! Get the microscopic cross sections for the given particle and material
    virtual MicroXsBuilders micro_xs(Applicability range) const = 0;

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::post; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
