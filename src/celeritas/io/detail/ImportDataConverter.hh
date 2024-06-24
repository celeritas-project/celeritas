//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/detail/ImportDataConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"

namespace celeritas
{
struct ImportData;
struct ImportElement;
struct ImportEmParameters;
struct ImportMaterial;
struct ImportModel;
struct ImportModelMaterial;
struct ImportMscModel;
struct ImportOpticalMaterial;
struct ImportParticle;
struct ImportPhysicsTable;
struct ImportProcess;
struct ImportGeoMaterial;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert imported data from one unit system to another.
 */
class ImportDataConverter
{
  public:
    // Construct with a unit system
    explicit ImportDataConverter(UnitSystem usys);

    //!@{
    //! Convert imported data to the native unit type
    void operator()(ImportData* data);
    void operator()(ImportElement* data);
    void operator()(ImportEmParameters* data);
    void operator()(ImportGeoMaterial* data);
    void operator()(ImportMaterial* data);
    void operator()(ImportOpticalMaterial* data);
    void operator()(ImportModel* data);
    void operator()(ImportModelMaterial* data);
    void operator()(ImportMscModel* data);
    void operator()(ImportParticle* data);
    void operator()(ImportPhysicsTable* data);
    void operator()(ImportProcess* data);
    //!@}

  private:
    UnitSystem usys_;
    double len_;
    double numdens_;
    double time_;
    double xs_;
    double inv_pressure_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
