#pragma once
#include <cstdint>
#include <algorithm>
#include <cassert>
#include "ring.hpp"
#include "pool.hpp"
#include "model.hpp"
#include "bitstream.hpp"
#include "rice.hpp"

namespace RLGR {
    constexpr uint32_t MAX_K = 240;
    constexpr uint32_t UP_K = 4;
    constexpr uint32_t DOWN_K = 6;
    constexpr uint32_t UP_K_DIRECT_RL = 3;
    constexpr uint32_t DOWN_K_DIRECT_RL = 3;
    //constexpr uint32_t UP_K_RICE = 2;
    constexpr uint32_t DOWN_K_RICE = 2;
    class Encoder { 
        uint32_t rl=0;
        uint32_t k=0;
        uint32_t kr=0;
#ifdef USE_MODEL_K
        ModelK model;
#endif
        bool inrun{false};
        BitStream::Writer s;     
        public:
        Encoder(PagePool& pool) : s(pool) {}
    
        void reserve(int res) {
            s.reserve(res); //syncpoint before
        }
    
        void put(uint32_t v) {
    
            uint32_t k_ = k >> 3;
            uint32_t kr_ = kr >> 3;
            uint32_t kr__ = kr_;                
            if (k_ > 0) { 
                    if (!inrun) {
                        reserve(1+k_);
                        inrun = true;
                    }
                if (v == 0) {                               
                   rl++; 
                   if (rl == (1u << k_)) {
                     assert(rl != 1);
#ifdef USE_MODEL_K
                      model.update(0);
#endif
                      s.put_bits(1,0);// полное окно
                      rl=0;
                      k += UP_K;
                      if (k > MAX_K)
                        k = MAX_K;
                      k_ = k >> 3;
                      //когда декодер вычитает этот бит полного окна - он обязать ждать rl-шагов прежде чем сделать этот резерв
                      reserve(1+k_);//k может вырастить тут - мы не будем ее считать - просто заразервируем 2 бита
                   }
                } else {        
                    inrun = false;  
                    s.put_bits(1,1);                
                    s.put_bits(k_,rl);
                    //тут уже нет резерва.
                    //тут можно сразу сказать декодеру сколько нужно брать из полубайты
                    //после того как он выдаст все свои нули. если декодер сразу прочитает 
                    //терминатор, то он не попадёт в своё окно. окно будет тут, а это будущее для декодера
                    
                    //std::cerr << "RL " << rl << std::endl;
               
                    k -= k > DOWN_K ? DOWN_K : k;
#ifdef USE_MODEL_K
                    if (kr__ <= 15 &&  model.trust()) {
                        kr__ = model.get_k();
                    } 
#endif
                    reserve(64);
                    rl=0; 
                    v--;
                    Rice::write(v,kr__,s);
#ifdef USE_MODEL_K
                    model.update( std::min(15, std20::bit_width(v)));
#endif
                    uint32_t vk = (v) >> kr_;
                    if (vk == 0) {
                        kr -= kr > DOWN_K_RICE ? DOWN_K_RICE : kr;
                    } else
                    if (vk != 1) {
                        kr += vk;
                        if (kr > MAX_K) 
                            kr = MAX_K;
                    }
                }
            } else {
                rl=0;
                reserve(64); 
#ifdef USE_MODEL_K
                if (kr__ <= 15 &&  model.trust()) {
                    kr__ = model.get_k();
                }
#endif
                Rice::write(v,kr__,s);            
                uint32_t vk = v >> kr_;
                if (vk == 0) {
                    kr -= kr > DOWN_K_RICE ? DOWN_K_RICE : kr;
                } else
                if (vk != 1) {
                    kr += vk;
                    if (kr > MAX_K)
                        kr = MAX_K;
                }
                if (v == 0) {
                    k += UP_K_DIRECT_RL;
                    if (k > MAX_K)
                        k = MAX_K;
                } else {
                    k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
                }
                k_ = k >> 3;
#ifdef USE_MODEL_K
                model.update( std::min(15, std20::bit_width(v)));
#endif
            }
        }
    
        void flush() {
            if (rl > 0) {
                uint32_t k_ = k>>3;    
                assert(k_ > 0);
                s.put_bits(1,1);
                s.put_bits(k_,rl);
            }
            s.flush();
        }
    };
    
    class Decoder {
    private:
        uint32_t rl = 0;
        bool run_mode{false};
        uint32_t k = 0;
        uint32_t kr = 0;
    #ifdef USE_MODEL_K   
        ModelK model;
    #endif
        bool fullrl{false};
    
        BitStream::Reader s;
    
    
        void reserve(int res) {
            s.reserve(res);
        }
    
        uint32_t readRL(bool terminator = false) {
            uint32_t kr_ = kr >> 3;
            uint32_t kr__ = kr_;
    #ifdef USE_MODEL_K
            if (kr__ <= 15 && model.trust()) {
                kr__ = model.get_k();
            }
    #endif
            auto rl_val = Rice::read(kr__, s);
            uint32_t vk = rl_val >> kr_;
            if (vk == 0) {
                kr -= kr > DOWN_K_RICE ? DOWN_K_RICE : kr;
            } else if (vk != 1) {
                kr += vk;
                if (kr > MAX_K)
                    kr = MAX_K;
            }
    #ifdef USE_MODEL_K
            model.update(std::min(15, std20::bit_width(rl_val)));
    #endif
            return rl_val + terminator;
        }
    
        // Логика завершения частичной серии (бывшая метка rl_0)
        uint32_t finish_partial_run() {
            reserve(64);
            k -= k > DOWN_K ? DOWN_K : k;
            auto t = readRL(true);
            run_mode = false;
            stream_size--;
            return t;
        }
    
        // Единый метод обработки шага внутри run_mode (бывшая метка rl_entry)
        uint32_t process_run() {
            if (fullrl) {
                assert(rl > 0);
                rl--;
                if (rl == 0) {
                    // Конец полного окна
                    k += UP_K;
                    if (k > MAX_K)
                        k = MAX_K;
                    auto k_ = k >> 3;
                    reserve(1 + k_);
                    run_mode = false;
                }
                stream_size--;
                return 0;
            } else {
                // Частичная серия
                if (rl == 0) {
                    return finish_partial_run();
                }
                assert(rl > 0);
                rl--;
                // Возвращаем 0, режим run_mode остается true
                return 0; 
            }
        }
    
        uint32_t read_direct() {
            reserve(64);
            uint32_t v = readRL();
            if (v == 0) {
                k += UP_K_DIRECT_RL;
                if (k > MAX_K)
                    k = MAX_K;
            } else {
                k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
            }
            stream_size--;
            return v;
        }
    
    public:
        size_t stream_size{0};
    
        Decoder(const  PagePool &pool, size_t stream_size = size_t(-1)) 
            : s(pool), stream_size(stream_size) {
        }
    
        uint32_t get() {
            // 1. Если уже в режиме серии - продолжаем (аналог прямого перехода на rl_entry)
            if (run_mode) {
                return process_run();
            }
    
            assert(stream_size > 0 && "read after eof");
            
            uint32_t k_ = k >> 3;
            
            // 2. Попытка входа в режим серии
            if (k_ > 0) {
                assert(rl == 0);
    
                // Оптимизация reserve из оригинала: делается только если fullrl был false
                // Важно: здесь используется значение fullrl от ПРЕДЫДУЩЕГО шага
                if (!fullrl) {
                    reserve(1 + k_);
                }
                
                // Читаем бит режима
                fullrl = s.peek_n(1) == 0;
                s.skip(1);
                run_mode = true;
    
                if (fullrl) {
    #ifdef USE_MODEL_K
                    model.update(0);
    #endif
                    rl = 1 << k_;
    
                } else {
                    rl = s.peek_n(k_);
                    s.skip(k_);
                    
                    // В оригинале: if (rl == 0) goto rl_0;
                    if (rl == 0) {
                        return finish_partial_run();
                    }
                    
    
                }
                // В оригинале: goto rl_entry;
                return process_run();
            }
    
            // 3. Прямое чтение (k=0)
            return read_direct();
        }
    };
}
    