//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Model.hh
//---------------------------------------------------------------------------//
#pragma once

#include <set>
#include <string>
#include "base/Span.hh"
#include "physics/grid/UniformGrid.hh"
#include "Applicability.hh"
#include "Types.hh"

namespace celeritas
{
struct ModelInteractPointers;
//---------------------------------------------------------------------------//
/*!
 * Abstract base class representing a physics model.
 *
 * A Model is a representation (often an approximation) to a physics process
 * such as Compton scattering that is valid for one or more particle types in a
 * given range (or ranges) of energy.
 *
 * Each Model subclass is constructed with a unique ModelId by a Process, which
 * is effectively a group of Models. Once constructed, it is essentially
 * immutable.
 *
 * The model assumes a few responsibilities:
 * - It provides accessors for the ranges of applicability: the same model
 *   (interaction kernel) can apply to multiple particles at different energy
 *   ranges.
 * - It precalculates macroscopic cross sections for each range of
 *   applicability.
 * - It precalculates energy loss rates and range limiters for each range.
 * - If it has an interaction cross section, it provides an "interact" method
 *   for undergoing an interaction and possibly emitting secondaries.
 *
 * This class is similar to Geant4's G4VContinuousDiscrete process, but more
 * limited.
 */
class Model
{
  public:
    //@{
    //! Type aliases
    using SetApplicability = std::set<Applicability>;
    //@}

  public:
    // Virtual destructor for polymorphic deletion
    virtual ~Model();

    //! Get the applicable particle type and energy ranges of the model
    virtual SetApplicability applicability() const = 0;

    //! Apply the interaction kernel for this model to all applicable tracks
    virtual void interact(const ModelInteractPointers&) const = 0;

    //! ID of the model (should be stored by constructor)
    virtual ModelId model_id() const = 0;

    //! Name of the model, for user interaction
    virtual std::string label() const = 0;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
