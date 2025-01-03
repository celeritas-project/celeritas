//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/EnergyLossTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "EnergyLossDeltaDistribution.hh"
#include "EnergyLossGammaDistribution.hh"
#include "EnergyLossGaussianDistribution.hh"
#include "EnergyLossHelper.hh"
#include "EnergyLossUrbanDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<EnergyLossFluctuationModel M>
struct EnergyLossTraits;

template<>
struct EnergyLossTraits<EnergyLossFluctuationModel::none>
{
    using type = EnergyLossDeltaDistribution;
};

template<>
struct EnergyLossTraits<EnergyLossFluctuationModel::gamma>
{
    using type = EnergyLossGammaDistribution;
};

template<>
struct EnergyLossTraits<EnergyLossFluctuationModel::gaussian>
{
    using type = EnergyLossGaussianDistribution;
};

template<>
struct EnergyLossTraits<EnergyLossFluctuationModel::urban>
{
    using type = EnergyLossUrbanDistribution;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
