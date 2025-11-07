//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
struct ImportPhysMaterial;
struct ImportModel;
struct ImportModelMaterial;
struct ImportMscModel;
struct ImportOpticalMaterial;
struct ImportOpticalModel;
struct ImportPhysicsTable;
struct ImportProcess;
struct ImportGeoMaterial;

namespace inp
{
struct Particle;
}  // namespace inp

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
    void operator()(ImportPhysMaterial* data);
    void operator()(ImportOpticalMaterial* data);
    void operator()(ImportOpticalModel* data);
    void operator()(ImportModel* data);
    void operator()(ImportModelMaterial* data);
    void operator()(ImportMscModel* data);
    void operator()(inp::Particle* data);
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
