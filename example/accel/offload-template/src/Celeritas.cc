//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/offload-template/src/Celeritas.cc
//---------------------------------------------------------------------------//
#include "Celeritas.hh"

#include <G4Threading.hh>
#include <accel/AlongStepFactory.hh>
#include <celeritas/alongstep/AlongStepGeneralLinearAction.hh>
#include <celeritas/alongstep/AlongStepUniformMscAction.hh>
#include <celeritas/em/params/UrbanMscParams.hh>
#include <celeritas/field/UniformFieldData.hh>
#include <celeritas/io/ImportData.hh>
#include <celeritas_version.h>

using namespace celeritas;

namespace
{
//---------------------------------------------------------------------------//
std::shared_ptr<CoreStepActionInterface const>
make_nofield_along_step(AlongStepFactoryInput const& input)
{
    CELER_LOG(debug) << "Creating along-step action with linear propagation";
    std::shared_ptr<AlongStepGeneralLinearAction> asgla;

    asgla = celeritas::AlongStepGeneralLinearAction::from_params(
        input.action_id,
        *input.material,
        *input.particle,
        celeritas::UrbanMscParams::from_import(
            *input.particle, *input.material, *input.imported),
        input.imported->em_params.energy_loss_fluct);

    return asgla;
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Globally shared setup options.
 * Setup options are constructed the first time this method is invoked.
 */
SetupOptions& CelerSetupOptions()
{
    static SetupOptions options = [] {
        // See celeritas/install/include/accel/SetupOptions.hh
        SetupOptions so;

        // Set along-step factory
        so.make_along_step = UniformAlongStepFactory();

        so.max_num_tracks = 1024 * 16;
        so.initializer_capacity = 1024 * 128 * 4;
        so.secondary_stack_factor = 3.0;
        so.ignore_processes = {"CoulombScat"};

        // Use Celeritas "hit processor" to call back to Geant4 SDs.
        so.sd.enabled = true;

        // Only call back for nonzero energy depositions: this is currently a
        // global option for all detectors, so if any SDs extract data from
        // tracks with no local energy deposition over the step, it must be set
        // to false.
        so.sd.ignore_zero_deposition = true;

        // Using the pre-step point, reconstruct the G4 touchable handle.
        so.sd.locate_touchable = true;
        // Reconstruct the track, needed for particle type
        so.sd.track = true;

        // Save diagnostic information
        so.output_file = "celeritas-offload-diagnostic.json";

        // Post-step data is used
        so.sd.pre.position = true;
        so.sd.pre.global_time = true;
        return so;
    }();
    return options;
}

//---------------------------------------------------------------------------//
/*!
 * Celeritas problem data.
 */
SharedParams& CelerSharedParams()
{
    static SharedParams sp;
    return sp;
}

//---------------------------------------------------------------------------//
/*!
 * Thread-local (when supported) transporter.
 */
LocalTransporter& CelerLocalTransporter()
{
    static G4ThreadLocal LocalTransporter lt;
    return lt;
}
