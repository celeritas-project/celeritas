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
 * MC truth data structures. Initial implementation keeps them as simple as
 * possible and should not rely on anything other than structs made of
 * primitive types (plus std::vectors), since these can be read directly by
 * ROOT without the need of dictionaries.
 *
 * `TStepData` and `TStepPoint` naming convention *must be* the same
 * as defined in `StepData.hh` so that the creation of branches can be managed
 * automatically (see details in \c RootStepWriter::make_tree() ).
 */

//---------------------------------------------------------------------------//
//! Pre- and post-step point information.
struct TStepPoint
{
    int    volume;
    double dir[3];
    double pos[3]; //!< [cm]
    double energy; //!< [MeV]
    double time;   //!< [s]
};

//---------------------------------------------------------------------------//
//! Full step data.
struct TStepData
{
    int        event;
    int        track;
    int        action;
    int        track_step_count;
    int        particle;          //!< PDG numbering scheme
    TStepPoint points[2];         //!< Pre- and post-step specific data
    double     energy_deposition; //!< [MeV]
    double     step_length;       //!< [cm]
};
//---------------------------------------------------------------------------//
} // namespace mctruth
//---------------------------------------------------------------------------//
} // namespace celeritas
