//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceMultiStepper.cc
//---------------------------------------------------------------------------//
#include "DormandPrinceMultiStepper.hh"

#if !CELER_USE_DEVICE
template<class E>
 auto
DormandPrinceMultiStepper<E>::operator()(real_type step,
                                    OdeState const& beg_state,
                                    int id, int index) const
    -> result_type
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif