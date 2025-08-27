//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/DebugIO.json.hh
//! \brief Write *on-host* track views to JSON for debugging.
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
class OrangeTrackView;

//---------------------------------------------------------------------------//
void to_json(nlohmann::json&, OrangeTrackView const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
