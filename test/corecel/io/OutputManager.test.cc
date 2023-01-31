//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputManager.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/OutputManager.hh"

#include <exception>
#include <regex>
#include <sstream>

#include "corecel/io/BuildOutput.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/sys/TypeDemangler.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class TestInterface final : public OutputInterface
{
  public:
    TestInterface(Category cat, std::string lab, int value)
        : cat_(cat), label_(lab), value_(value)
    {
    }

    Category category() const final { return cat_; }
    std::string label() const final { return label_; }

    void output(JsonPimpl* json) const final
    {
#if CELERITAS_USE_JSON
        json->obj = value_;
#else
        (void)sizeof(json);
        (void)sizeof(value_);
#endif
    }

  private:
    Category cat_{};
    std::string label_{};
    int value_{};
};

//---------------------------------------------------------------------------//
// **IMPORTANT** this class cannot be `final` for exception nesting to work!
// Its members can be, though.
class MockKernelContextException : public RichContextException
{
  public:
    MockKernelContextException(int th, int ev, int tr)
        : thread_(th), event_(ev), track_(tr)
    {
    }

    char const* type() const final { return "MockKernelContextException"; }

    void output(JsonPimpl* json) const final
    {
#if CELERITAS_USE_JSON
        json->obj["thread"] = thread_;
        json->obj["event"] = event_;
        json->obj["track"] = track_;
#else
        (void)sizeof(json);
#endif
    }

  private:
    int thread_{};
    int event_{};
    int track_{};
};

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class OutputManagerTest : public Test
{
  protected:
    using Category = OutputInterface::Category;

    std::string to_string(OutputManager const& om)
    {
        static const std::regex file_match(R"re("file":"[^"]+")re");
        static const std::regex line_match(R"re("line":[0-9]+)re");
        std::ostringstream os;
        om.output(&os);
        std::string result = os.str();
        result = std::regex_replace(result, file_match, R"("file":"FILE")");
        result = std::regex_replace(result, line_match, R"("line":123)");
        return result;
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(OutputManagerTest, empty)
{
    OutputManager om;

    std::string result = this->to_string(om);
    if (CELERITAS_USE_JSON)
    {
        EXPECT_EQ("null", result);
    }
    else
    {
        EXPECT_EQ("\"output unavailable\"", result);
    }
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

    EXPECT_THROW(om.insert(first), RuntimeError);

    std::string result = this->to_string(om);
    if (CELERITAS_USE_JSON)
    {
        EXPECT_EQ(
            R"json({"input":{"input_value":42},"result":{"out":1,"timing":2}})json",
            result);
    }
    else
    {
        EXPECT_EQ("\"output unavailable\"", result);
    }
}

TEST_F(OutputManagerTest, build_output)
{
    OutputManager om;
    om.insert(std::make_shared<celeritas::BuildOutput>());
    std::string result = this->to_string(om);
    EXPECT_TRUE(result.find("CELERITAS_BUILD_TYPE") != std::string::npos)
        << "actual output: " << result;
}

TEST_F(OutputManagerTest, exception_output)
{
    OutputManager om;
    auto exception_to_output = [&om](std::exception_ptr const& ep) {
        om.insert(std::make_shared<celeritas::ExceptionOutput>(ep));
    };

    CELER_TRY_HANDLE(
        CELER_VALIDATE(false, << "things went wrong"),
        exception_to_output);

    std::string result = this->to_string(om);
    if (CELERITAS_USE_JSON)
    {
        EXPECT_EQ(R"json({"result":{"exception":{"condition":"false","file":"FILE","line":123,"type":"RuntimeError","what":"things went wrong","which":"runtime"}}})json", result);
    }
}

TEST_F(OutputManagerTest, nested_exception_output)
{
    OutputManager om;
    auto exception_to_output = [&om](std::exception_ptr const& ep) {
        om.insert(std::make_shared<celeritas::ExceptionOutput>(ep));
    };

    CELER_TRY_HANDLE_CONTEXT(CELER_VALIDATE(false, << "things went wrong"),
                             exception_to_output,
                             MockKernelContextException(123, 2, 4567));


    std::string result = this->to_string(om);
    if (CELERITAS_USE_JSON)
    {
        EXPECT_EQ(R"json({"result":{"exception":{"condition":"false","context":{"event":2,"thread":123,"track":4567,"type":"MockKernelContextException"},"file":"FILE","line":123,"type":"RuntimeError","what":"things went wrong","which":"runtime"}}})json", result);

    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
