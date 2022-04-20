//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OutputManager.test.cc
//---------------------------------------------------------------------------//
#include "sim/OutputManager.hh"

#include <sstream>

#include "sim/JsonPimpl.hh"

#include "celeritas_test.hh"

using celeritas::JsonPimpl;
using celeritas::OutputManager;

class TestInterface final : public celeritas::OutputInterface
{
  public:
    TestInterface(Category cat, std::string lab, int value)
        : cat_(cat), label_(lab), value_(value)
    {
    }

    Category    category() const final { return cat_; }
    std::string label() const final { return label_; }

    void output(JsonPimpl* json) const final
    {
#if CELERITAS_USE_JSON
        json->obj = value_;
#else
        (void)sizeof(json);
#endif
    }

  private:
    Category    cat_{};
    std::string label_{};
    int         value_{};
};

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class OutputManagerTest : public celeritas::Test
{
  protected:
    using Category = celeritas::OutputInterface::Category;

    std::string to_string(const OutputManager& om)
    {
        std::ostringstream os;
        om.output(&os);
        return os.str();
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(OutputManagerTest, empty)
{
    OutputManager om;

    std::string result = this->to_string(om);
#if CELERITAS_USE_JSON
    EXPECT_EQ("null", result);
#else
    EXPECT_EQ("\"output unavailable\"", result);
#endif
}

TEST_F(OutputManagerTest, minimal)
{
    auto first
        = std::make_shared<TestInterface>(Category::input, "input_value", 42);
    auto second = std::make_shared<TestInterface>(Category::result, "out", 1);
    auto third = std::make_shared<TestInterface>(Category::result, "timing", 2);

    OutputManager om;
    om.insert(first);
    om.insert(second);
    om.insert(third);

    EXPECT_THROW(om.insert(first), celeritas::RuntimeError);

    std::string result = this->to_string(om);
#if CELERITAS_USE_JSON
    EXPECT_EQ(R"({"input":{"input_value":42},"result":{"out":1,"timing":2}})",
              result);
#else
    EXPECT_EQ("\"output unavailable\"", result);
#endif
}
