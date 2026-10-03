//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file orange/OrangeShift.test.cu
 * \note This is compiled by the CUDA compiler but doesn't contain any
 * executable code. It is here to ensure that no CUDA-breaking code
 * ends up in header files picked up by SCALE.
 */
//---------------------------------------------------------------------------//

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/sys/Device.hh"
#include "orange/OrangeInput.hh"
#include "orange/OrangeParams.hh"
#include "orange/OrangeTrackView.hh"
#include "orange/OrangeTypes.hh"
