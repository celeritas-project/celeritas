//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MpiTypes.hh
//! Type definitions for MPI parallel operations
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

#if CELERITAS_USE_MPI
#    include "./MpiTypes.mpi.i.hh"
#else
#    include "./MpiTypes.nompi.i.hh"
#endif
