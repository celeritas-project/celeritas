//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ApplyCutoffProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/Types.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
class CutoffParams;
}

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Debugging class to kill particles.
 *
 * With a cross section of 1e100 for all materials for the given particle
 * types, absorb the particle.
 */
class ApplyCutoffProcess : public celeritas::Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstCutoff = std::shared_ptr<const celeritas::CutoffParams>;
    //!@}

  public:
    // Construct from cutoffs
    explicit ApplyCutoffProcess(SPConstCutoff cutoffs);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability applic) const final;

    //! Type of process
    ProcessType type() const final
    {
        return ProcessType::electromagnetic_discrete;
    }

    //! Name of the process
    std::string label() const final { return "ApplyCutoff absorption"; }

  private:
    SPConstCutoff cutoffs_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
