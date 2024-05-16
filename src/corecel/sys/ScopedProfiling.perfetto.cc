//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.perfetto.cc
//! \brief The roctx implementation of \c ScopedProfiling
//---------------------------------------------------------------------------//
#include "ScopedProfiling.hh"

namespace celeritas
{
void initialize_perfetto(perfetto::TracingInitArgs const& args)
{
    perfetto::Tracing::Initialize(args);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas