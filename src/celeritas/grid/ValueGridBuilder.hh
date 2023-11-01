//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ValueGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
class ValueGridInserter;
struct XsGridData;

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
    //! \name Type aliases
    using ValueGridId = ItemId<struct XsGridData>;
    //!@}

  public:
    //! Virtual destructor for polymorphic deletion
    virtual ~ValueGridBuilder() = 0;

    //! Construct the grid given a mutable reference to a store
    virtual ValueGridId build(ValueGridInserter) const = 0;

  protected:
    ValueGridBuilder() = default;

    //!@{
    //! Prevent copy/move except by daughters that know what they're doing
    ValueGridBuilder(ValueGridBuilder const&) = default;
    ValueGridBuilder& operator=(ValueGridBuilder const&) = default;
    ValueGridBuilder(ValueGridBuilder&&) = default;
    ValueGridBuilder& operator=(ValueGridBuilder&&) = default;
    //!@}
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
    //! \name Type aliases
    using SpanConstDbl = Span<double const>;
    using VecDbl = std::vector<double>;
    //!@}

  public:
    // Construct from imported data
    static std::unique_ptr<ValueGridXsBuilder>
    from_geant(SpanConstDbl lambda_energy,
               SpanConstDbl lambda,
               SpanConstDbl lambda_prim_energy,
               SpanConstDbl lambda_prim);

    // Construct from just scaled cross sections
    static std::unique_ptr<ValueGridXsBuilder>
    from_scaled(SpanConstDbl lambda_prim_energy, SpanConstDbl lambda_prim);

    // Construct
    ValueGridXsBuilder(double emin, double eprime, double emax, VecDbl xs);

    // Construct in the given store
    ValueGridId build(ValueGridInserter) const final;

  private:
    double log_emin_;
    double log_eprime_;
    double log_emax_;
    VecDbl xs_;
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
    //! \name Type aliases
    using VecDbl = std::vector<double>;
    using SpanConstDbl = Span<double const>;
    using Id = ItemId<XsGridData>;
    using UPLogBuilder = std::unique_ptr<ValueGridLogBuilder>;
    //!@}

  public:
    // Construct from full grids
    static UPLogBuilder from_geant(SpanConstDbl energy, SpanConstDbl value);

    // Construct from range
    static UPLogBuilder from_range(SpanConstDbl energy, SpanConstDbl range);

    // Construct
    ValueGridLogBuilder(double emin, double emax, VecDbl value);

    // Construct in the given store
    ValueGridId build(ValueGridInserter) const final;

    // Access values
    SpanConstDbl value() const;

  private:
    double log_emin_;
    double log_emax_;
    VecDbl value_;
};

//---------------------------------------------------------------------------//
/*!
 * Build a physics vector for quantities that are nonuniform in energy.
 */
class ValueGridGenericBuilder final : public ValueGridBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using VecDbl = std::vector<double>;
    using Id = ItemId<XsGridData>;
    //!@}

  public:
    // Construct
    ValueGridGenericBuilder(VecDbl grid,
                            VecDbl value,
                            Interp grid_interp,
                            Interp value_interp);

    // Construct with linear interpolation
    ValueGridGenericBuilder(VecDbl grid, VecDbl value);

    // Construct in the given store
    ValueGridId build(ValueGridInserter) const final;

  private:
    VecDbl grid_;
    VecDbl value_;
    Interp grid_interp_;
    Interp value_interp_;
};

//---------------------------------------------------------------------------//
/*!
 * Special cases for indicating *only* on-the-fly cross sections.
 *
 * Currently this should be thrown just for processes and models specified in
 * \c HardwiredModels as needed for EPlusAnnihilationProcess, which has *only*
 * on-the-fly cross section calculation.
 *
 * This class is needed so that the process has at least one "builder"; but it
 * always returns an invalid ValueGridId.
 */
class ValueGridOTFBuilder final : public ValueGridBuilder
{
  public:
    // Don't construct anything
    ValueGridId build(ValueGridInserter) const final;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
