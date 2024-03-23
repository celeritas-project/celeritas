//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "celeritas/Types.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "ImportedOpticalProcessAdapter.hh"

namespace celeritas
{
class OpticalModel;
//---------------------------------------------------------------------------//
/*!
 * Base class for optical processes.
 *
 * In general, an optical process is uniquely identified with its optical
 * model since there's only one applicable energy range for optical photons.
 * Therefore, unlike in particle process classes, only one model is ever
 * built. The helper template OpticalProcessImpl is used to define a unique
 * OpticalProcess for each OpticalModel.
 *
 * Also unlike particle processes, optical photons use GenericGrids for
 * linear energy interpolation instead of log interpolation.
 * StepLimitBuilders cannot be built for OpticalProcesses as they would for
 * particle processes, so a GenericGridInserter is passed used to build
 * the grids instead.
 */
class OpticalProcess
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstModel = std::shared_ptr<OpticalModel const>;
    using SPConstImported = std::shared_ptr<ImportOpticalProcesses const>;
    using ActionIdIter = RangeIter<ActionId>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    //!@}

  public:
    
    //! Construct the optical process
    inline CELER_FUNCTION OpticalProcess(ImportOpticalProcessClass ipc,
                                         SPConstMaterials materials,
                                         SPConstImported shared_data);

    //! Get the interaction cross sections for optical photons
    std::vector<OpticalValueGridId>
    step_limits(GenericGridInserter& inserter) const;

    //! Build the corresponding OpticalModel with the given action id
    virtual SPConstModel build_model(ActionIdIter start_id) const = 0;

    //! Label of the optical process
    virtual std::string label() const = 0;

  protected:
    ImportedOpticalProcessAdapter imported_;
    SPConstMaterials materials_;
};

//---------------------------------------------------------------------------//
/*!
 * Helper template to identify a given OpticalModel with a unique
 * OpticalProcess.
 *
 * The OpticalModel is meant to implement the physics and data specific to
 * each optical process, and this template is a quick way to build processes
 * without having to rewrite more OpticalProcess classes. Static methods in
 * OpticalModel subclasses are used to specificy the process parameters.
 *
 * It should be called with a simple using statement after the model class
 * is defined.
 * \code
    using AbsorptionProcess = OpticalProcessImpl<AbsorptionModel>;
   \endcode
 */
template <class OpticalModelImpl>
class OpticalProcessInstance : public OpticalProcess
{
  public:
    //! Construct the optical process.
    inline CELER_FUNCTION
    OpticalProcessInstance(SPConstMaterials materials,
                           SPConstImported shared_data)
        : OpticalProcess(OpticalModelImpl::process_class(), materials, shared_data)
    {}

    //! Create unique model associated with this process
    SPConstModel build_model(ActionIdIter start_id) const override final
    {
        return std::make_shared<OpticalModelImpl>(*start_id++, *materials_);
    }

    //! Name of the optical process
    std::string label() const override final
    {
        return OpticalModelImpl::process_label();
    }
};

using AbsorptionProcess = OpticalProcessInstance<class AbsorptionModel>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct the optical process from the given IPC.
 */
OpticalProcess::OpticalProcess(ImportOpticalProcessClass ipc,
                               SPConstMaterials materials,
                               SPConstImported shared_data)
    : imported_(shared_data, ipc), materials_(std::move(materials))
{
    CELER_EXPECT(materials_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the step limits and add them to the given grid inserter.
 * Returns a list of grid ids added.
 */
std::vector<OpticalValueGridId>
OpticalProcess::step_limits(GenericGridInserter& inserter) const
{
    return imported_.step_limits(inserter, *materials_);
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
