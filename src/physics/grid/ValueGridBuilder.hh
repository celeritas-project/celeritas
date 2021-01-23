//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/Span.hh"
#include "base/Types.hh"

namespace celeritas
{
class ValueGridStore;
//---------------------------------------------------------------------------//
//! Parameterization of the energy grid values for a physics array
enum class EnergyLookup
{
    uniform_log, //!< Uniform in log(E), interpolate in log(E)
};

//---------------------------------------------------------------------------//
//! Parameterization of the value calculation
enum class ValueCalculation
{
    linear,        //!< Linear interpolation in value
    linear_scaled, //!< Linear interpolation, then divide by energy above E'
};

//---------------------------------------------------------------------------//
/*!
 * Helper class for constructing on-device physics data for a single material.
 *
 * The physics manager will assemble all the array builders during setup, will
 * query all of them for the storage types and requirements, and allocate the
 * necessary storage. Each instance of the physics array builder class will
 * then copy the class data to the device.
 *
 * These builder classes are presumed to have a short/temporary lifespan and
 * should not be retained after the setup phase.
 */
class ValueGridBuilder
{
  public:
    //!@{
    //! Type aliases
    using EnergyStorage = std::pair<EnergyLookup, size_type>;
    using ValueStorage  = std::pair<ValueCalculation, size_type>;
    //!@}

  public:
    virtual ~ValueGridBuilder() = 0;

    virtual EnergyStorage energy_storage() const       = 0;
    virtual ValueStorage  value_storage() const        = 0;
    virtual void          build(ValueGridStore*) const = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Build a physics array for EM process cross sections.
 *
 * This array has a uniform grid in log(E) and a special value above which the
 * input cross sections are scaled by E.
 */
class ValueGridXsBuilder final : public ValueGridBuilder
{
  public:
    //!@{
    //! Type aliases
    using SpanConstReal = Span<const real_type>;
    //!@}

  public:
    // Construct from imported data
    static ValueGridXsBuilder from_geant(SpanConstReal lambda_energy,
                                         SpanConstReal lambda,
                                         SpanConstReal lambda_prim_energy,
                                         SpanConstReal lambda_prim);

    // Construct
    ValueGridXsBuilder(real_type              emin,
                       real_type              eprime,
                       real_type              emax,
                       std::vector<real_type> xs);

    // Get the storage type and requirements for the energy grid.
    EnergyStorage energy_storage() const final;

    // Get the storage type and requirements for the value grid.
    ValueStorage value_storage() const final;

    void build(ValueGridStore*) const final;

  private:
    real_type              log_emin_;
    real_type              log_eprime_;
    real_type              log_emax_;
    std::vector<real_type> xs_;
};

// TODO: implement ValueGridLogBuilder
#if 0
//---------------------------------------------------------------------------//
/*!
 * Build a physics vector for energy loss and other quantities.
 *
 * This vector has a uniform grid in log(E).
 */
class ValueGridLogBuilder final : public ValueGridBuilder
{
};
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
