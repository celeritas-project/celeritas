//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceFwd.hh
//! \brief Forward declaration of surface classes
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<Axis T>
class ConeAligned;
template<Axis T>
class CylAligned;
template<Axis T>
class CylCentered;
class GeneralQuadric;
class Plane;
template<Axis T>
class PlaneAligned;
class SimpleQuadric;
class Sphere;
class SphereCentered;

//---------------------------------------------------------------------------//
}  // namespace celeritas
