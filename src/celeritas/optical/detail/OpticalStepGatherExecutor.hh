//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalStepGatherExecutor.hh
//---------------------------------------------------------------------------//
#pragma once
#include <fstream>
#include <nlohmann/json.hpp>

#include "celeritas/Types.hh"
#include "celeritas/optical/CoreTrackView.hh"

#include "OpticalStepData.hh"
#include "OpticalStepParams.hh"

namespace celeritas
{
namespace optical
{

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    OpticalStepGatherExecutor ...;
   \endcode
 */
template<StepPoint P>
struct OpticalStepGatherExecutor
{
    NativeCRef<CoreParamsData> const& params;
    NativeRef<CoreStateData> const& states;

    NativeRef<OpticalStepStateData> step;

    CELER_FUNCTION void operator()(CoreTrackView& track) const
    {
        // CoreTrackView track(params, states, tid);

        if (!is_track_valid(track.sim().status()))
            return;

        // if constexpr (P == StepPoint::pre)
        // {
        //  CELER_LOG(info) << "=====Prestep position======";
        //
        //  auto tid = track.track_slot_id();
        //  auto idx = tid.get();
        //  auto geo = track.geometry();
        //  auto sim = track.sim();
        //  auto mat = track.material_record();
        //  auto part = track.particle();
        //  auto surf = track.surface();
        //  auto const& pos = geo.pos();
        //  auto const& direction = geo.dir();
        //
        //  CELER_LOG(info)
        //      << "Volume name in the optical: "
        //      << track.geometry().volume_id().unchecked_get()
        //      << " at x = " << pos[0] << "y = " << pos[1]
        //      << " z = " << pos[2] << " energy = " << part.energy().value()
        //      << " material id " << mat.material_id().unchecked_get()
        //      << " Post step action id :  "
        //      << sim.post_step_action().unchecked_get() << " Direction : x "
        //      << direction[0] << "y = " << direction[1]
        //      << " z = " << direction[2];
        // }
        if constexpr (P == StepPoint::post)
        {
            auto tid = track.track_slot_id();
            auto idx = tid.get();
            auto geo = track.geometry();
            auto sim = track.sim();
            auto mat = track.material_record();
            auto part = track.particle();
            auto surf = track.surface();
            auto const& pos = geo.pos();
            auto const& direction = geo.dir();

            static std::ofstream out("optical_steps.jsonl", std::ios::app);

            nlohmann::json j;
            j["slot"] = idx;
            j["volume"] = geo.volume_id().unchecked_get();
            j["x"] = pos[0];
            j["y"] = pos[1];
            j["z"] = pos[2];
            j["energy"] = part.energy().value();
            j["dir_x"] = direction[0];
            j["dir_y"] = direction[1];
            j["dir_z"] = direction[2];
            j["material"] = mat.material_id().unchecked_get();
            j["post_action"] = sim.post_step_action().unchecked_get();

            out << j.dump() << "\n";

            //  step.track_slot[tid] = tid;
            //  step.pos[tid] = geo.pos()
            //  step.volume_id[tid] = geo.volume_id();
            //  CELER_LOG(info) << "Exec";
            //  if (idx >= step.track_slot.size())
            //  {
            //      CELER_LOG(error) << "OUT OF BOUNDS: tid=" << idx
            //                       << " step.size=" <<
            //                       step.track_slot.size();
            //      return;
            //  }

            // step.track_slot[tid] = tid;
            // step.pos[tid] = track.geometry().pos();
            // step.volume_id[tid] = track.geometry().volume_id();
            CELER_LOG(info)
                << "Volume name in the optical: "
                << track.geometry().volume_id().unchecked_get()
                << " at x = " << pos[0] << "y = " << pos[1]
                << " z = " << pos[2] << " energy = " << part.energy().value()
                << " material id "
                << mat.material_id().unchecked_get()
                //  << " Surface id :" << surf_phys.material().unchecked_get()
                << " Post step action id :  "
                << sim.post_step_action().unchecked_get();
            auto id = sim.post_step_action();
        }
    }
};

}  // namespace detail
}  // namespace optical
}  // namespace celeritas
