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
auto DormandPrinceMultiStepper<E>::operator()(real_type step,
                                              OdeState const& beg_state,
                                              int id,
                                              int index,
                                              OdeState* ks,
                                              OdeState* along_state,
                                              FieldStepperResult* result) const

    template<class E>
    auto DormandPrinceMultiStepper<E>::run_sequential(
        real_type step,
        OdeState const& beg_state,
        int id,
        int mask,
        OdeState* ks,
        OdeState* along_state,
        FieldStepperResult* result) const

    template<class E>
    auto DormandPrinceMultiStepper<E>::run_aside(
        real_type step,
        OdeState const& beg_state,
        int id,
        int index,
        int mask,
        OdeState* ks,
        OdeState* along_state,
        FieldStepperResult* result) const;
#endif