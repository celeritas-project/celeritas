//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/detail/GridAccessor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

#include "../NonuniformGridData.hh"
#include "../UniformGrid.hh"
#include "../UniformGridData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for accessing grid data in different formats.
 */
class GridAccessor
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstReal = Span<real_type const>;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Virtual destructor for polymorphic deletion
    virtual ~GridAccessor() = default;

    //! Get the x grid value at the given index
    virtual real_type x(size_type index) const = 0;

    //! Get the y grid value at the given index
    virtual real_type y(size_type index) const = 0;

    //! Get the grid size
    virtual size_type size() const = 0;

    // Calculate \f$ \Delta x_i = x_{i + 1} - x_i \f$
    inline real_type delta_x(size_type index) const;

    // Calculate \f$ \Delta y_i = y_{i + 1} - y_i \f$
    inline real_type delta_y(size_type index) const;

    // Calculate \f$ r_i - r_{i - 1} \f$
    inline real_type delta_slope(size_type index) const;
};

//---------------------------------------------------------------------------//
/*!
 * Helper class for accessing grid data from a nonuniform grid.
 */
class NonuniformGridAccessor : public GridAccessor
{
  public:
    // Construct with spans
    inline NonuniformGridAccessor(SpanConstReal x_values,
                                  SpanConstReal y_values);

    // Construct with a nonuniform grid
    inline NonuniformGridAccessor(NonuniformGridRecord const& grid,
                                  Values const& values);

    // Get the x grid value at the given index
    inline real_type x(size_type index) const final;

    // Get the y grid value at the given index
    inline real_type y(size_type index) const final;

    //! Get the grid size
    size_type size() const final { return x_values_.size(); }

  private:
    SpanConstReal x_values_;
    SpanConstReal y_values_;
};

//---------------------------------------------------------------------------//
/*!
 * Helper class for accessing grid data from a uniform grid.
 */
class UniformGridAccessor : public GridAccessor
{
  public:
    // Construct with grid data
    inline UniformGridAccessor(UniformGridRecord const& grid,
                               Values const& values);

    //! Get the x grid value at the given index
    inline real_type x(size_type index) const final;

    //! Get the y grid value at the given index
    inline real_type y(size_type index) const final;

    //! Get the grid size
    size_type size() const final { return loge_grid_.size(); }

  private:
    UniformGridRecord const& data_;
    Values const& reals_;
    UniformGrid loge_grid_;
};

//---------------------------------------------------------------------------//
/*!
 * Helper class for accessing grid data from an inverse uniform grid.
 */
class InverseGridAccessor : public GridAccessor
{
  public:
    //!@{
    //! \name Type aliases
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct with grid data
    inline InverseGridAccessor(UniformGridRecord const& grid,
                               Values const& values);

    //! Get the x grid value at the given index
    inline real_type x(size_type index) const final;

    //! Get the y grid value at the given index
    inline real_type y(size_type index) const final;

    //! Get the grid size
    size_type size() const final { return loge_grid_.size(); }

  private:
    UniformGridRecord const& data_;
    Values const& reals_;
    UniformGrid loge_grid_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate \f$ x_{i + 1} - x_i \f$.
 */
real_type GridAccessor::delta_x(size_type index) const
{
    return this->x(index + 1) - this->x(index);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate \f$ y_{i + 1} - y_i \f$
 */
real_type GridAccessor::delta_y(size_type index) const
{
    return this->y(index + 1) - this->y(index);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate \f$ r_i - r_{i - 1} \f$.
 *
 * This calculates \f$ \Delta r_i = \frac{\Delta y_i}{\Delta x_i} -
 * \frac{\Delta y_{i - 1}}{\Delta x_{i - 1}} \f$
 */
real_type GridAccessor::delta_slope(size_type index) const
{
    CELER_EXPECT(index > 0);
    return this->delta_y(index) / this->delta_x(index)
           - this->delta_y(index - 1) / this->delta_x(index - 1);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from spans.
 */
NonuniformGridAccessor::NonuniformGridAccessor(SpanConstReal x_values,
                                               SpanConstReal y_values)
    : x_values_(x_values), y_values_(y_values)
{
    CELER_EXPECT(x_values_.size() == y_values_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a nonuniform grid.
 */
NonuniformGridAccessor::NonuniformGridAccessor(NonuniformGridRecord const& grid,
                                               Values const& values)

    : NonuniformGridAccessor(values[grid.grid], values[grid.value])
{
}

//---------------------------------------------------------------------------//
/*!
 * Get the x grid value at the given index.
 */
real_type NonuniformGridAccessor::x(size_type index) const
{
    CELER_EXPECT(index < this->size());
    return x_values_[index];
}

//---------------------------------------------------------------------------//
/*!
 * Get the y grid value at the given index.
 */
real_type NonuniformGridAccessor::y(size_type index) const
{
    CELER_EXPECT(index < this->size());
    return y_values_[index];
}

//---------------------------------------------------------------------------//
/*!
 * Construct from cross section grid.
 */
UniformGridAccessor::UniformGridAccessor(UniformGridRecord const& grid,
                                         Values const& values)
    : data_(grid), reals_(values), loge_grid_(data_.grid)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the x grid value at the given index.
 */
real_type UniformGridAccessor::x(size_type index) const
{
    CELER_EXPECT(index < this->size());
    return std::exp(loge_grid_[index]);
}

//---------------------------------------------------------------------------//
/*!
 * Get the y grid value at the given index.
 */
real_type UniformGridAccessor::y(size_type index) const
{
    CELER_EXPECT(index < this->size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
/*!
 * Construct from cross section grid.
 */
InverseGridAccessor::InverseGridAccessor(UniformGridRecord const& grid,
                                         Values const& values)
    : data_(grid), reals_(values), loge_grid_(data_.grid)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the x grid value at the given index.
 */
real_type InverseGridAccessor::x(size_type index) const
{
    CELER_EXPECT(index < this->size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
/*!
 * Get the y grid value at the given index.
 */
real_type InverseGridAccessor::y(size_type index) const
{
    CELER_EXPECT(index < this->size());
    return std::exp(loge_grid_[index]);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
