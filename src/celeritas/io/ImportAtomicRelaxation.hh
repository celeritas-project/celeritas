//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportAtomicRelaxation.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * EADL transition data for atomic relaxation for a single element.
 */
struct ImportAtomicTransition
{
    int initial_shell{};  //!< Originating shell designator
    int auger_shell{};  //!< Auger shell designator
    double probability{};  //!< Transition probability
    double energy{};  //!< Transition energy [MeV]
};

struct ImportAtomicSubshell
{
    int designator{};  //!< Subshell designator
    std::vector<ImportAtomicTransition> fluor;  //!< Radiative transitions
    std::vector<ImportAtomicTransition> auger;  //!< Non-radiative transitions
};

struct ImportAtomicRelaxation
{
    std::vector<ImportAtomicSubshell> shells;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
