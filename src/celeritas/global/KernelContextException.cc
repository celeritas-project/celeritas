//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/KernelContextException.cc
//---------------------------------------------------------------------------//
#include "KernelContextException.hh"

#include "corecel/OpaqueIdIO.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/math/QuantityIO.json.hh"
#include "corecel/sys/Environment.hh"

#include "CoreTrackView.hh"
#include "Debug.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<class V, class S>
void insert_if_valid(char const* key,
                     OpaqueId<V, S> const& val,
                     nlohmann::json* obj)
{
    if (val)
    {
        (*obj)[key] = val.unchecked_get();
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with track data and kernel label.
 */
KernelContextException::KernelContextException(
    HostCRef<CoreParamsData> const& params,
    HostRef<CoreStateData> const& states,
    ThreadId thread,
    std::string_view label)
    : thread_(thread), label_{label}
{
    try
    {
        if (!thread)
        {
            // Make sure the thread is valid before trying to construct
            // detailed debug information
            throw std::exception();
        }
        CoreTrackView core(params, states, thread);
        this->initialize(core);
    }
    catch (...)
    {
        // Ignore all exceptions while trying to process diagnostic information
        what_ = label_ + " (error processing track state)";
    }
}

//---------------------------------------------------------------------------//
/*!
 * This class type's description.
 */
char const* KernelContextException::type() const
{
    return "KernelContextException";
}

//---------------------------------------------------------------------------//
/*!
 * Save context to a JSON object.
 */
void KernelContextException::output(JsonPimpl* json) const
{
    nlohmann::json j;
#define KCE_INSERT_IF_VALID(ATTR) insert_if_valid(#ATTR, ATTR##_, &j)

    KCE_INSERT_IF_VALID(thread);
    KCE_INSERT_IF_VALID(track_slot);
    KCE_INSERT_IF_VALID(event);
    KCE_INSERT_IF_VALID(track);
    if (track_)
    {
        KCE_INSERT_IF_VALID(parent);
        j["num_steps"] = num_steps_;
        KCE_INSERT_IF_VALID(particle);
        j["energy"] = energy_;
        j["pos"] = pos_;
        j["dir"] = dir_;
        KCE_INSERT_IF_VALID(volume);
        KCE_INSERT_IF_VALID(surface);
    }
    if (!label_.empty())
    {
        j["label"] = label_;
    }
#undef KCE_INSERT_IF_VALID
    json->obj = std::move(j);
}

//---------------------------------------------------------------------------//
/*!
 * Populate properties during construction.
 */
void KernelContextException::initialize(CoreTrackView const& core)
{
    track_slot_ = core.track_slot_id();
    auto const&& sim = core.sim();
    if (sim.status() != TrackStatus::inactive)
    {
        event_ = sim.event_id();
        track_ = sim.track_id();
        parent_ = sim.parent_id();
        num_steps_ = sim.num_steps();
        {
            auto const&& par = core.particle();
            particle_ = par.particle_id();
            energy_ = par.energy();
        }
        {
            auto const&& geo = core.geometry();
            pos_ = geo.pos();
            dir_ = geo.dir();
            if (!geo.is_outside())
            {
                volume_ = geo.volume_id();
            }
            surface_ = geo.internal_surface_id();
        }
    }
    {
        // Construct std::exception message
        std::ostringstream os;
        os << "track slot " << track_slot_ << " in kernel '" << label_ << "'";
        if (track_)
        {
            os << ": " << StreamableTrack{core};
        }
        what_ = os.str();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
