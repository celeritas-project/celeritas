//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/SimEnergyDepositData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Required input data to construct a \c sim::SimEnergyDeposit class object.
 *
 * This struct is used to store/read the required data to construct a \c
 * sim::SimEnergyDeposit object into/from a ROOT file without the need of a
 * dictionary.
 *
 * The struct element names replicate the \c sim::SimEnergyDeposit getters for
 * easier manipulation using C++ macros.
 *
 * The data members are pointers to allow the reuse of this struct by both the
 * exporter ( \c GeoSimExporterModule ) and reader ( \c LarDataReader )
 * classes: For the reader, the pointers are necessary, as ROOT \em requires
 * \c nullptr during \c TTree::SetBranchAddress .
 *
 * \sa GeoSimExporterModule
 * \sa LarDataReader
 */
struct SimEnergyDepositData
{
    std::vector<int>* NumPhotons{nullptr};
    std::vector<int>* NumElectrons{nullptr};
    std::vector<double>* ScintYieldRatio{nullptr};
    std::vector<double>* Energy{nullptr};
    std::vector<double>* Time{nullptr};
    std::vector<double>* StartX{nullptr};
    std::vector<double>* StartY{nullptr};
    std::vector<double>* StartZ{nullptr};
    std::vector<double>* EndX{nullptr};
    std::vector<double>* EndY{nullptr};
    std::vector<double>* EndZ{nullptr};
    std::vector<double>* StartT{nullptr};
    std::vector<double>* EndT{nullptr};
    std::vector<int>* TrackID{nullptr};
    std::vector<int>* PdgCode{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
