//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/MCTruthData.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace mctruth
{
//---------------------------------------------------------------------------//
/*!
 * MCtruth data structures. Initial implementation is to keep them as simple as
 * possible and do not rely on anything other than structs made of primitive
 * types (plus std::vector), since these can be read directly by ROOT without
 * the need of dictionaries.
 */

//---------------------------------------------------------------------------//
//! Pre- and post-step point information.
struct TStepPoint
{
    int    volume_id;
    double dir[3];
    double pos[3]; //!< [cm]
    double energy; //!< [MeV]
    double time;   //!< [s]
};

//---------------------------------------------------------------------------//
//! Full step data.
struct TStepData
{
    int        event_id;
    int        track_id;
    int        action_id;
    int        pdg;
    int        track_step_count;
    TStepPoint points[2];         // Pre- and post-step specific data
    double     energy_deposition; //!< [MeV]
    double     length;            //!< [cm]
};
//---------------------------------------------------------------------------//
} // namespace mctruth
//---------------------------------------------------------------------------//
} // namespace celeritas
