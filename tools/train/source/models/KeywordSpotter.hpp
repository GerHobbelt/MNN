#ifndef KEYWORD_SPOT_HPP
#define KEYWORD_SPOT_HPP

#include <MNN/expr/Module.hpp>

namespace MNN {
    namespace Train {
        namespace Model {
            class MNN_PUBLIC KeywordSpotter : public Express::Module {
                public:
                    KeywordSpotter();

                    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;

                    std::shared_ptr<Express::Module> conv1;
                    std::shared_ptr<Express::Module> conv2;
                    std::shared_ptr<Express::Module> fc1;
                    std::shared_ptr<Express::Module> dropout;
            }; // class KeywordSpotter
        } // namespace Model
    } // namespace Train
} // namespace MNN

#endif // KEYWORD_SPOT_HPP