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
#include "base/Collection.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"

namespace celeritas
{
class ValueGridInserter;
//---------------------------------------------------------------------------//
/*!
 * Helper class for constructing on-device physics data for a single material.
 *
 * These builder classes are presumed to have a short/temporary lifespan and
 * should not be retained after the setup phase.
 */
class ValueGridBuilder
{
  public:
    //!@{
    //! Type aliases
    using ValueGridId = ItemId<struct XsGridData>;
    //!@}

  public:
    //! Virtual destructor for polymorphic deletion
    virtual ~ValueGridBuilder() = 0;

    //! Construct the grid given a mutable reference to a store
    virtual ValueGridId build(ValueGridInserter) const = 0;
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
    static std::unique_ptr<ValueGridXsBuilder>
    from_geant(SpanConstReal lambda_energy,
               SpanConstReal lambda,
               SpanConstReal lambda_prim_energy,
               SpanConstReal lambda_prim);

    // Construct
    ValueGridXsBuilder(real_type emin,
                       real_type eprime,
                       real_type emax,
                       VecReal   xs);

    // Construct in the given store
    ValueGridId build(ValueGridInserter) const final;

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
class ValueGridLogBuilder : public ValueGridBuilder
{
  public:
    //!@{
    //! Type aliases
    using VecReal       = std::vector<real_type>;
    using SpanConstReal = Span<const real_type>;
    using Id            = ItemId<XsGridData>;
    //!@}

  public:
    // Construct
    ValueGridLogBuilder(real_type emin, real_type emax, VecReal value);

    // Construct in the given store
    ValueGridId build(ValueGridInserter) const final;

    //! Access Values
    SpanConstReal value() const { return make_span(value_); }

  private:
    real_type log_emin_;
    real_type log_emax_;
    VecReal   value_;
};

//---------------------------------------------------------------------------//
/*!
 * Build a physics vector for range tables.
 *
 * Range tables are uniform in log(E), and range must monotonically increase
 * with energy.
 */
class ValueGridRangeBuilder : public ValueGridLogBuilder
{
    using Base = ValueGridLogBuilder;

  public:
    // Construct
    ValueGridRangeBuilder(real_type emin, real_type emax, VecReal value);
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
    using Id      = ItemId<XsGridData>;
    //!@}

  public:
    // Construct
    ValueGridGenericBuilder(VecReal grid,
                            VecReal value,
                            Interp  grid_interp,
                            Interp  value_interp);

    // Construct with linear interpolation
    ValueGridGenericBuilder(VecReal grid, VecReal value);

    // Construct in the given store
    ValueGridId build(ValueGridInserter) const final;

  private:
    VecReal grid_;
    VecReal value_;
    Interp  grid_interp_;
    Interp  value_interp_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
