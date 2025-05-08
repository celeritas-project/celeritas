//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Process.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>  // IWYU pragma: export
#include <vector>

#include "corecel/cont/Range.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/Grid.hh"

namespace celeritas
{
struct Applicability;
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
 *
 * \todo energy loss should use a \c UniformGrid
 */
class Process
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstModel = std::shared_ptr<Model const>;
    using VecModel = std::vector<SPConstModel>;
    using ActionIdIter = RangeIter<ActionId>;
    using XsGrid = inp::XsGrid;
    using EnergyLossGrid = inp::XsGrid;
    //!@}

  public:
    // Virtual destructor for polymorphic deletion
    virtual ~Process();

    //! Construct the models associated with this process
    virtual VecModel build_models(ActionIdIter start_id) const = 0;

    //! Get the interaction cross sections [l/len] for the given energy range
    virtual XsGrid macro_xs(Applicability range) const = 0;

    //! Get the energy loss [MeV/len] for the given energy range
    virtual EnergyLossGrid energy_loss(Applicability range) const = 0;

    //! Whether the integral method can be used to sample interaction length
    virtual bool supports_integral_xs() const = 0;

    //! Whether the process applies when the particle is stopped
    virtual bool applies_at_rest() const = 0;

    //! Name of the process
    virtual std::string_view label() const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    Process() = default;
    CELER_DEFAULT_COPY_MOVE(Process);
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
