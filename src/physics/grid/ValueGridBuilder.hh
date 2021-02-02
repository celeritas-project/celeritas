//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include "base/Span.hh"
#include "base/Types.hh"

namespace celeritas
{
class ValueGridStore;
//---------------------------------------------------------------------------//
//! Parameterization of the energy grid values for a physics array
enum class ValueGridType
{
    xs,
    generic
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
    using Storage = std::pair<ValueGridType, size_type>;
    //!@}

  public:
    //! Virtual destructor for polymorphic deletion
    virtual ~ValueGridBuilder() = 0;

    //! Get the storage requirements of a grid to be bulit
    virtual Storage storage() const = 0;

    //! Construct the grid given a mutable reference to a store
    virtual void build(ValueGridStore*) const = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Build a physics array for EM process cross sections.
 *
 * This array has a uniform grid in log(E) and a special value at or above
 * which the input cross sections are scaled by E.
 */
class ValueGridXsBuilder final : public ValueGridBuilder
{
  public:
    //!@{
    //! Type aliases
    using SpanConstReal = Span<const real_type>;
    using VecReal       = std::vector<real_type>;
    //!@}

  public:
    // Construct from imported data
    static ValueGridXsBuilder from_geant(SpanConstReal lambda_energy,
                                         SpanConstReal lambda,
                                         SpanConstReal lambda_prim_energy,
                                         SpanConstReal lambda_prim);

    // Construct
    ValueGridXsBuilder(real_type emin,
                       real_type eprime,
                       real_type emax,
                       VecReal   xs);

    // Get the storage type and requirements
    Storage storage() const final;

    // Construct in the given store
    void build(ValueGridStore*) const final;

  private:
    real_type log_emin_;
    real_type log_eprime_;
    real_type log_emax_;
    VecReal   xs_;
};

//---------------------------------------------------------------------------//
/*!
 * Build a physics vector for energy loss and other quantities.
 *
 * This vector is still uniform in log(E).
 */
class ValueGridLogBuilder final : public ValueGridBuilder
{
  public:
    //!@{
    //! Type aliases
    using VecReal = std::vector<real_type>;
    //!@}

  public:
    // Construct
    ValueGridLogBuilder(real_type emin, real_type emax, VecReal value);

    // Get the storage type and requirements
    Storage storage() const final;

    // Construct in the given store
    void build(ValueGridStore*) const final;

  private:
    real_type log_emin_;
    real_type log_emax_;
    VecReal   xs_;
};

//---------------------------------------------------------------------------//
/*!
 * Build a physics vector for quantities that are nonuniform in energy.
 */
class ValueGridGenericBuilder final : public ValueGridBuilder
{
  public:
    //!@{
    //! Type aliases
    using VecReal = std::vector<real_type>;
    //!@}

  public:
    // Construct
    ValueGridGenericBuilder(VecReal grid,
                            VecReal value,
                            Interp  grid_interp,
                            Interp  value_interp);

    // Construct with linear interpolation
    ValueGridGenericBuilder(VecReal grid, VecReal value);

    // Get the storage type and requirements
    Storage storage() const final;

    // Construct in the given store
    void build(ValueGridStore*) const final;

  private:
    VecReal grid_;
    VecReal value_;
    Interp  grid_interp_;
    Interp  value_interp_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
