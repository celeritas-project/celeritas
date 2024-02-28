//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgObject.hh
//! \brief CSG operations on Object instances: negation, union, intersection
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "ObjectInterface.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Everywhere *but* the embedded object.
 */
class NegatedObject : public ObjectInterface
{
  public:
    // Construct with the object to negate and an empty name
    explicit NegatedObject(SPConstObject obj);

    // Construct with a name and object
    NegatedObject(std::string&& label, SPConstObject obj);

    //! Access the daughter object
    SPConstObject const& daughter() const { return obj_; }

    //! Get the user-provided label
    std::string_view label() const final { return label_; }

    // Construct a volume from this object
    NodeId build(VolumeBuilder&) const final;

    // Write the object to JSON
    void output(JsonPimpl*) const final;

  private:
    std::string label_;
    SPConstObject obj_;
};

//---------------------------------------------------------------------------//
/*!
 * Join all of the given objects with an intersection or union.
 */
template<OperatorToken Op>
class JoinObjects : public ObjectInterface
{
    static_assert(Op == op_and || Op == op_or);

  public:
    //!@{
    //! \name Type aliases
    using VecObject = std::vector<SPConstObject>;
    //!@}

    //! Operation joining the daughters ("and" or "or")
    static constexpr OperatorToken op_token = Op;

  public:
    // Construct with a label and vector of objects
    JoinObjects(std::string&& label, VecObject&& objects);

    //! Access the vector of daughter objects
    VecObject const& daughters() const { return objects_; }

    //! Get the user-provided label
    std::string_view label() const final { return label_; }

    // Construct a volume from this object
    NodeId build(VolumeBuilder&) const final;

    // Write the object to JSON
    void output(JsonPimpl*) const final;

  private:
    std::string label_;
    VecObject objects_;
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Union of the given objects
using AnyObjects = JoinObjects<op_or>;
//! Intersection of the given objects
using AllObjects = JoinObjects<op_and>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Make a new object that is the second object subtracted from the first
std::shared_ptr<AllObjects const>
make_subtraction(std::string&& label,
                 std::shared_ptr<ObjectInterface const> const& minuend,
                 std::shared_ptr<ObjectInterface const> const& subtrahend);

// Make a combination of possibly negated objects
std::shared_ptr<AllObjects const> make_rdv(
    std::string&& label,
    std::vector<std::pair<Sense, std::shared_ptr<ObjectInterface const>>>&&);

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
