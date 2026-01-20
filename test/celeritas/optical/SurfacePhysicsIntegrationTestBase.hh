//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsIntegrationTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "corecel/data/AuxStateVec.hh"
#include "corecel/math/Turn.hh"
#include "corecel/sys/ActionGroups.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/Transporter.hh"
#include "celeritas/optical/gen/DirectGeneratorAction.hh"
#include "celeritas/optical/surface/SurfacePhysicsParams.hh"
#include "celeritas/track/TrackFunctors.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
/*!
 * Reference results:
 * - Double precision
 * - Orange geometry (requires valid surface normals and relocation on
 *   boundary)
 */
constexpr bool reference_configuration
    = ((CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
       && (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
       && (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW));

//---------------------------------------------------------------------------//
/*!
 * Template class for capturing photons after a surface interaction and scoring
 * them with the given functor.
 */
template<class Collector>
class CollectResultsAction final : public OpticalStepActionInterface,
                                   public ConcreteAction
{
  public:
    explicit CollectResultsAction(ActionId aid, Collector& results)
        : ConcreteAction(aid, "collect-results", "collect test results")
        , results_(results)
    {
    }

    void step(CoreParams const& params, CoreStateHost& state) const final
    {
        for (auto tid : range(TrackSlotId{state.size()}))
        {
            CoreTrackView track(params.host_ref(), state.ref(), tid);
            auto sim = track.sim();
            if (this->is_post_boundary(track)
                || this->is_absorbed_on_boundary(track))
            {
                results_(track);
                sim.status(TrackStatus::killed);
            }
        }
    }

    void step(CoreParams const&, CoreStateDevice&) const final
    {
        CELER_NOT_IMPLEMENTED("not collecting on device");
    }

    StepActionOrder order() const final { return StepActionOrder::post; }

  private:
    //! Whether the track finished a boundary crossing
    inline bool is_post_boundary(CoreTrackView const& track) const
    {
        return AppliesValid{}(track)
               && track.sim().post_step_action()
                      == track.surface_physics().scalars().post_boundary_action;
    }

    //! Whether the track was absorbed during a boundary crossing
    inline bool is_absorbed_on_boundary(CoreTrackView const& track) const
    {
        return track.sim().status() == TrackStatus::killed
               && track.sim().post_step_action()
                      == track.surface_physics().scalars().surface_stepping_action;
    }

    Collector& results_;
};

//---------------------------------------------------------------------------//
/*!
 * A test base for running surface physics integration tests.
 *
 * Tests are run in the optical-box.gdml setup, where photons are initialized
 * close to the top (positive-y) edge and are shot directly into it. The
 * collect action is used to capture photons immediately after a surface
 * interaction and log them in an appropriate functor.
 */
class SurfacePhysicsIntegrationTestBase
    : public ::celeritas::test::GeantTestBase
{
  public:
    std::string_view gdml_basename() const override { return "optical-box"; }

    GeantPhysicsOptions build_geant_options() const override;
    GeantImportDataSelection build_import_data_selection() const override;
    std::vector<IMC> select_optical_models() const override;
    SPConstOpticalSurfacePhysics build_optical_surface_physics() override;

    //! Initialize transporter and state for run
    void initialize_run();

    //! Run a single set of photons at the given angle
    void run_step(RealTurn angle);

    //! Create a collector action for the given functor
    template<class C>
    void create_collector(C& collect)
    {
        auto& reg = *this->optical_params()->action_reg();
        auto collector = std::make_shared<CollectResultsAction<C>>(
            reg.next_id(), collect);
        reg.insert(collector);
    }

  protected:
    std::shared_ptr<CoreState<MemSpace::host>> state_;
    std::shared_ptr<AuxStateVec> aux_;
    std::shared_ptr<Transporter> transport_;
    std::shared_ptr<DirectGeneratorAction> generate_;

    virtual void setup_surface_models(inp::SurfacePhysics&) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
