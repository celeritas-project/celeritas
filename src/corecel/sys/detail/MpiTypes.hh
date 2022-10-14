//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/MpiTypes.hh
//! Type definitions for MPI parallel operations (implementation-agnostic)
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

#if CELERITAS_USE_MPI
#    include "./MpiTypes.mpi.hh"
#else
#    include "./MpiTypes.nompi.hh"
#endif
