//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"
#if CELERITAS_VECGEOM_VERSION >= 0x020000
#    include "VecgeomTrackView.v2.hh"
#else
#    include "VecgeomTrackView.v1.hh"
#endif
