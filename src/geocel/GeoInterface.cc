//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoInterface.cc
//---------------------------------------------------------------------------//
#include "GeoParamsInterface.hh"
#include "GeoTrackInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Default virtual destructor
GeoParamsInterface::~GeoParamsInterface() = default;

//---------------------------------------------------------------------------//
//! Default virtual destructor
template<class RealType>
GeoTrackInterface<RealType>::~GeoTrackInterface() = default;

#if CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
template class GeoTrackInterface<float>;
#endif
template class GeoTrackInterface<double>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
